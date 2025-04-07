# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:54:33 2024

@author: carll
"""

# Fit models to Simulated Experiment 2 Data - Variable Feedback
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from multiprocessing import Pool
from scipy.stats import truncnorm
from scipy.optimize import minimize
import math
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import TimeSeriesSplit

#----------------------------------------------------------------------------
# Utility functions
#----------------------------------------------------------------------------

def add_session_column(df, condition_col='condition'):

    """
    Adds a 'session' column to the DataFrame. The session number increments
    each time the value in the condition_col changes from 'neut', 'pos',
    or 'neg' to 'baseline'.

    :param df: Pandas DataFrame to which the session column will be added
    :param condition_col: Name of the column containing the condition values
    :return: DataFrame with the added 'session' column
    """

    session = 0
    session_numbers = []
    previous_condition = None  # Variable to store the previous condition

    for _, row in df.iterrows():
        current_condition = row[condition_col]
        if (previous_condition in ['neut', 'pos', 'neg']
                and current_condition == 'baseline'):
            session += 1
        session_numbers.append(session)
        previous_condition = current_condition

    df_copy = df.copy()
    df_copy['session'] = session_numbers

    return df_copy

def calc_norm_prob_vectorized(confidence, mean, sigma,
                              lower_bound=0, upper_bound=100):
    """
    Calculate the probability of confidence values given a normal distribution
    with corresponding mean and sigma values for each point. Ensure that the
    probability density values for integer steps between lower_bound and upper_bound
    sum to 1 for each distribution, using vectorized operations.

    Parameters:
    confidence (array): List or array of confidence values.
    mean (array): List or array of mean values corresponding to each confidence value.
    sigma (array): List or array of standard deviation values corresponding to each confidence value.
    lower_bound (int): The lower bound for the range of values to normalize over (default 0).
    upper_bound (int): The upper bound for the range of values to normalize over (default 100).

    Returns:
    numpy.ndarray: An array of probabilities for each confidence[i] based on
                   the normalized normal distribution defined by mean[i] and sigma[i].
    """

    # Create an array of integer steps [lower_bound, upper_bound] for all distributions
    x_values = np.arange(lower_bound, upper_bound + 1)

    # Reshape mean and sigma to broadcast across x_values (for element-wise operations)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]

    # Compute the Gaussian probability for each x_value, mean, and sigma (vectorized)
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)

    # Normalize each distribution by dividing each by the sum of the distribution
    y_values /= np.sum(y_values, axis=1, keepdims=True)

    # Ensure confidence is an array of integers
    confidence = confidence.astype(int)

    # Extract probabilities for the confidence values (vectorized)
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]

    return probabilities, y_values

def calc_inverted_norm_prob_vectorized(confidence, mean, sigma,
                                       lower_bound=0, upper_bound=100):
    """Same function as above, but now returns the
    inverted and normalized y_values"""
    # Create an array of integer steps
    x_values = np.arange(lower_bound, upper_bound + 1)

    # Reshape mean and sigma to broadcast across x_values
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]

    # Compute the Gaussian
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)

    # Invert
    y_values = np.max(y_values, axis=1, keepdims=True) - y_values

    # Normalize
    y_values /= np.sum(y_values, axis=1, keepdims=True)

    # Ensure confidence is an array of integers
    confidence = confidence.astype(int)

    # Extract probabilities for the confidence values
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]

    return probabilities, y_values

#----------------------------------------------------------------------------
# Models
#----------------------------------------------------------------------------

def RWP_trial (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the performance
    on the current trial (C_P).

    return list of vectors
    """

    alpha, sigma, bias, w_RW, w_P = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias

        else:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Encure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance influence
            c_P = performance[t] + 100

            # Get the prediction as the weighted sum
            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_P*c_P))))

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]

def RWP (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the performance
    on the current trial (C_P).
    """

    alpha, sigma, bias, w_RW, w_P = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias
        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance influence
            c_P = performance[t] + 100

            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_P*c_P))))

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return np.sum(nlls[trial_index])


def rwpd_trial (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the change in performance
    since the last trial (delta P).

    return list of vectors
    """

    alpha, sigma, bias, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias

        else:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Encure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get delta performance
            #delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between -100 and 100.
            #delta_p = max(-100, min(100, delta_p))

            # Get performance influence
            c_P =  max(0, min(100, w_PD*performance[t] + 100))

            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_PD*c_P))))

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]

def rwpd (x, *args):

    """
    Confidence is updated as the weighted sum of: (1)
    confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
    the change in performance since the last trial (delta P).
    """

    alpha, sigma, bias, w_RW, w_PD = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias
        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance delta
            #delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between -100 and 100.
            #delta_p = max(-100, min(100, delta_p))

            # Get performance influence
            # +100 is there to translate the truncated performance error into
            # confidence (0-100).
            c_P =  max(0, min(100, w_PD*performance[t] + 100))

            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_PD*c_P))))

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return np.sum(nlls[trial_index])

def RW_choice_kernel_trial(x, *args):

    """
     Models the evolution of confidence estimate probability over trials
     based on previously choosen confidence levels and a Rescorla-Wagner
     update rule. It calculates the sum of negativelog-likelihoods (NLL)
     for observed choices, given model parameters.

     Parameters:
     - x: Array containing model parameters:
         - alpha: Learning rate for updating choice kernels.
         - sigma: Standard deviation for Gaussian smoothing.
         - bias: Initial bias affecting starting probabilities.
         - beta: Scaling factor for the influence of choice kernels.
     - args: Tuple containing:
         - feedback: array of feedback
         - confidence: Array of confidence levels for each trial.
         - n_trials: Total number of trials.

     Process:
     1. Initializes choice kernels for a range of 101 possible choices.
     2. Iterates over trials, adjusting choice probabilities based on
        previous choices and confidence, using parameters in x.
     3. Applies Gaussian smoothing and normalization to model probabilities.
     4. Calculates NLL for actual choices.
     5. Return list of vectors of NLL, Sigma

     Returns:
     - Sum of NLLs for all trials, excluding the initial baseline, as a
       measure of model fit to the observed data.
    """

    alpha, alpha_ck, sigma, sigma_ck, bias, beta, beta_ck = x
    feedback, confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)
    model_pred = np.zeros(n_trials)
    model_probs = []
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:

            # Get starting probabilities
            t1_probs = np.full(array_length, value)

            # Set model prediction to bias
            model_pred[t] = bias

            # Make bias the most likely
            t1_probs[int(bias)] = 1

            # Smooth distribution
            t1_probs = gaussian_filter1d(t1_probs, sigma)

            # Normalize to get probabilities
            probabilities = t1_probs / np.sum(t1_probs)
            model_probs.append(probabilities)

            # Add to nll vector
            nlls[t] = -np.log(probabilities[int(confidence[t])])

        if t > 0:

            # Update confidence according to RW updates -----------------------
            # Get previous confidence estimate and feedback
            f = feedback[t-1]
            c = model_pred[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))

            # Calculate probabilities across different options (p_options)
            probability, p_options = calc_norm_prob_vectorized(
                                                    np.array([confidence[t]]),
                                                    np.array([c_pred]),
                                                    np.array([sigma]),
                                                    lower_bound=0,
                                                    upper_bound=100)
            # -----------------------------------------------------------------
            # Update choice kernels
            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha_ck * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels,
                                                        sigma_ck)

            # -----------------------------------------------------------
            # Combine RW update and CK update
            p = np.zeros(len(p_options[0]))
            for i, (v_k, ck_k) in enumerate(zip(p_options[0],
                                                smoothed_choice_kernels)):

                # Exponential of each beta * value + beta + choice kernel
                p[i] = np.exp((beta*v_k) + (beta_ck*ck_k))

            # Normalize to get probabilities
            probabilities = p / np.sum(p)
            model_probs.append(probabilities)

            # Get nll
            nlls[t] = -np.log(probabilities[int(confidence[t])] + epsilon)

    # Remove initial trial when the bias was applied
    nlls = nlls[1:]
    confidence = confidence[1:]
    sigma_vec = np.full(len(nlls), sigma)
    model_probs = model_probs[1:]

    return [nlls, model_probs, sigma_vec, confidence]


def RW_choice_kernel(x, *args):

    """
     Models the evolution of confidence estimate probability over trials
     based on previously choosen confidence levels and a Rescorla-Wagner
     update rule. It calculates the sum of negativelog-likelihoods (NLL)
     for observed choices, given model parameters.

     Parameters:
     - x: Array containing model parameters:
         - alpha: Learning rate for updating choice kernels.
         - sigma: Standard deviation for Gaussian smoothing.
         - bias: Initial bias affecting starting probabilities.
         - beta: Scaling factor for the influence of choice kernels.
     - args: Tuple containing:
         - feedback: array of feedback
         - confidence: Array of confidence levels for each trial.
         - n_trials: Total number of trials.

     Process:
     1. Initializes choice kernels for a range of 101 possible choices.
     2. Iterates over trials, adjusting choice probabilities based on
        previous choices and confidence, using parameters in x.
     3. Applies Gaussian smoothing and normalization to model probabilities.
     4. Calculates NLL for actual choices, summing across trials.

     Returns:
     - Sum of NLLs for all trials, excluding the initial baseline, as a
       measure of model fit to the observed data.
    """

    alpha, alpha_ck, sigma, sigma_ck, bias, beta, beta_ck = x
    feedback, confidence, n_trials, trial_index = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)
    model_pred = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:

            # Get starting probabilities
            t1_probs = np.full(array_length, value)

            # Make bias the most likely
            t1_probs[int(bias)] = 1

            # Smooth distribution
            t1_probs = gaussian_filter1d(t1_probs, sigma)

            # Normalize to get probabilities
            probabilities = t1_probs / np.sum(t1_probs)

            # Add to nll vector
            nlls[t] = -np.log(probabilities[int(confidence[t])])

        if t > 0:
            # Update "value" --------------------------------------------

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = model_pred[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

            # Calculate probabilities across different options (p_options)
            probability, p_options = calc_norm_prob_vectorized(
                                                    np.array([confidence[t]]),
                                                    np.array([c_pred]),
                                                    np.array([sigma]),
                                                    lower_bound=0,
                                                    upper_bound=100)

            # -----------------------------------------------------------

            # Update choice kernels
            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha_ck * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels,
                                                        sigma_ck)

            # -----------------------------------------------------------

            # Combine RW update and CK update
            p = np.zeros(len(p_options[0]))
            for i, (v_k, ck_k) in enumerate(zip(p_options[0],
                                                smoothed_choice_kernels)):

                # Exponential of each beta * value + beta + choice kernel
                p[i] = np.exp((beta*v_k) + (beta_ck*ck_k))

            # Normalize to get probabilities
            probabilities = p / np.sum(p)

            # Get nll
            nlls[t] = -np.log(probabilities[int(confidence[t])] + epsilon)

    # Remove initial trial
    nlls = nlls[1:]

    return np.sum(nlls[trial_index])


def choice_kernel_trial(x, *args):
    """
     Models the evolution of confidence estimate probability over trials
     based on previously choosen confidence levels. It calculates the sum of
     negativelog-likelihoods (NLL) for observed choices, given model
     parameters.

     Parameters:
     - x: Array containing model parameters:
         - alpha: Learning rate for updating choice kernels.
         - sigma: Standard deviation for Gaussian smoothing.
         - bias: Initial bias affecting starting probabilities.
         - beta: Scaling factor for the influence of choice kernels.
     - args: Tuple containing:
         - confidence: Array of confidence levels for each trial.
         - n_trials: Total number of trials.

     Process:
     1. Initializes choice kernels for a range of 101 possible choices.
     2. Iterates over trials, adjusting choice probabilities based on
        previous choices and confidence, using parameters in x.
     3. Applies Gaussian smoothing and normalization to model probabilities.
     4. Calculates NLL for actual choices, summing across trials.

     Returns:
     - list of vectors, including NLLs, sigma and confidence for all trials,
       excluding the initial baseline.
    """


    alpha, sigma, bias, beta = x
    confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)
    model_pred = np.zeros(n_trials)
    model_probs = []
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:

            # Get starting probabilities
            t1_probs = np.full(array_length, value)

            # Make bias the most likely
            t1_probs[int(bias)] = 1

            # Set model prediction to bias
            model_pred[t] = bias

            # Smooth distribution
            t1_probs = gaussian_filter1d(t1_probs, sigma)

            # Normalize to get probabilities
            probabilities = t1_probs / np.sum(t1_probs)

            model_probs.append(probabilities)

            # Add to nll vector
            nlls[t] = -np.log(probabilities[int(confidence[t])] + epsilon)

        if t > 0:

            # Get previous confidence value,
            c = int(confidence[t-1])

            # Update choice kernels
            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma)

            # Exponential of each choice kernel value
            exp_choice_kernels = np.exp(beta*smoothed_choice_kernels)

            # Normalize to get probabilities
            probabilities = exp_choice_kernels / np.sum(exp_choice_kernels)
            model_probs.append(probabilities)

            # Get nll
            nlls[t] = -np.log(probabilities[int(confidence[t])] + epsilon)

    # Remove initial trial
    nlls = nlls[1:]
    model_pred = model_pred[1:]
    model_probs = model_probs[1:]
    confidence = confidence[1:]

    # Sigma always the same
    sigma_vec = np.full(len(nlls), sigma)

    return [nlls, model_probs, sigma_vec, confidence]


def choice_kernel(x, *args):

    """
     Models the evolution of confidence estimate probability over trials
     based on previously choosen confidence levels. It calculates the sum of
     negativelog-likelihoods (NLL) for observed choices, given model
     parameters.

     Parameters:
     - x: Array containing model parameters:
         - alpha: Learning rate for updating choice kernels.
         - sigma: Standard deviation for Gaussian smoothing.
         - bias: Initial bias affecting starting probabilities.
         - beta: Scaling factor for the influence of choice kernels.
     - args: Tuple containing:
         - confidence: Array of confidence levels for each trial.
         - n_trials: Total number of trials.

     Process:
     1. Initializes choice kernels for a range of 101 possible choices.
     2. Iterates over trials, adjusting choice probabilities based on
        previous choices and confidence, using parameters in x.
     3. Applies Gaussian smoothing and normalization to model probabilities.
     4. Calculates NLL for actual choices, summing across trials.

     Returns:
     - Sum of NLLs for all trials, excluding the initial baseline, as a
       measure of model fit to the observed data.
    """

    alpha, sigma, bias, beta = x
    confidence, n_trials, trial_index = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)

    # Small value to avoid division by 0
    epsilon = 1e-10

    for t in range(n_trials):

        if t == 0:

            # Get starting probabilities
            t1_probs = np.full(array_length, value)

            # Make bias the most likely
            t1_probs[int(bias)] = 1

            # Smooth distribution
            t1_probs = gaussian_filter1d(t1_probs, sigma)

            # Normalize to get probabilities
            probabilities = t1_probs / np.sum(t1_probs)

            # Add to nll vector
            nlls[t] = -np.log(probabilities[int(confidence[t])] + epsilon)

        if t > 0:

            # Get previous confidence value
            c = int(confidence[t-1])

            # Update choice kernels
            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma)

            # Exponential of each choice kernel value
            exp_choice_kernels = np.exp(beta*smoothed_choice_kernels)

            # Normalize to get probabilities
            probabilities = exp_choice_kernels / np.sum(exp_choice_kernels)

            # Get nll
            nlls[t] = -np.log(probabilities[int(confidence[t])] + epsilon)

    # Remove initial trial
    nlls = nlls[1:]

    return np.sum(nlls[trial_index])


def rw_cond_LR_trial(x, *args):
    """
    Implements the Rescorla-Wagner model with one learning rate per feedback-
    condition. The function updates the confidence level
    based on feedback and the learning rate, calculating the negative
    log likelihood (NLL) for the model's predictions.

    Parameters:
    x (tuple): Contains learning rates (alpha_neut, alpha_pos, and alpha_neg)
    bias, and standard deviation (sigma).
    args (tuple): Includes arrays of confidence levels, feedback, and
                  the number of trials.

    Returns:
    list: The NLL, model predictions, standard deviation vector, and
          adjusted confidence levels for each trial.

    The function starts with a baseline trial and updates confidence
    levels in subsequent trials based on feedback. It ensures updated
    confidence levels remain within 0-100. The NLL is calculated using a
    truncated normal distribution to account for these bounds.The function
    ignores the baseline trial in its final output.
    """

    alpha_neut, alpha_pos, alpha_neg, sigma, bias = x
    confidence, feedback, n_trials, condition = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):


        if t == 0:
            # first trial: the mean for the first trial is set to bias
            model_pred[t] = bias

        if t > 0:

            # The feedback condition on the previous trial dictates the alpha
            if condition[t-1] == 'neut':
                alpha = alpha_neut
            if condition[t-1] == 'pos':
                alpha = alpha_pos
            if condition[t-1] == 'neg':
                alpha = alpha_neg

            # Get previous confidence estimate and feedback
            f = feedback[t-1]
            c = model_pred[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial trial
    confidence = confidence[1:]
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]

    # Calculate model prediction probabilities (probability)
    # and probability across all options (p_options)
    probability, p_options = calc_norm_prob_vectorized(confidence,
                                                       model_pred,
                                                       sigma_vec,
                                                       lower_bound=0,
                                                       upper_bound=100)
    # Get nll
    nlls = -np.log(probability + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]

def rw_cond_LR(x, *args):

    """
    Calculates the sum of negative log likelihood (NLL) for a sequence of
    trials using the Rescorla-Wagner model with symmetric learning rates.
    The model updates confidence levels based on prediction errors (PE),
    which are derived from the difference between feedback and prior
    confidence.

    Parameters:
    x (tuple): Contains alpha (learning rate) and sigma (standard deviation
               for the likelihood calculations).
    args (tuple): Includes arrays for confidence levels, feedback, and the
                  total number of trials.

    Returns:
    float: The total NLL for all trials, indicating the model's fit to
           the observed data.

    The function processes each trial, updating confidence levels based
    on feedback. It ensures updated confidence levels remain within 0-100.
    NLLs are calculated using a truncated normal distribution, adjusting
    for the bounded nature of confidence levels. The function ignores the
    baseline trial in its final output.
    """

    alpha_neut, alpha_pos, alpha_neg, sigma, bias = x
    confidence, feedback, n_trials, condition, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = bias

        if t > 0:

            # The feedback condition on the previous trial dictates the alpha
            if condition[t-1] == 'neut':
                alpha = alpha_neut
            if condition[t-1] == 'pos':
                alpha = alpha_pos
            if condition[t-1] == 'neg':
                alpha = alpha_neg

            # Get previous confidence estimate and feedback
            f = feedback[t-1]
            c = model_pred[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return np.sum(nlls[trial_index])


def rw_symmetric_LR_trial(x, *args):
    """
    Implements the Rescorla-Wagner model with symmetric learning rates
    for a series of trials. The function updates the confidence level
    based on feedback and the learning rate, calculating the negative
    log likelihood (NLL) for the model's predictions.

    Parameters:
    x (tuple): Contains learning rate (alpha) and standard deviation (sigma).
    args (tuple): Includes arrays of confidence levels, feedback, and
                  the number of trials.

    Returns:
    list: The NLL, model predictions, standard deviation vector, and
          adjusted confidence levels for each trial.

    The function starts with a baseline trial and updates confidence
    levels in subsequent trials based on feedback. It ensures updated
    confidence levels remain within 0-100. The NLL is calculated using a
    truncated normal distribution to account for these bounds.The function
    ignores the baseline trial in its final output.
    """

    alpha, sigma, bias = x
    confidence, feedback, n_trials = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = bias

        if t > 0:

            # Get previous confidence estimate and feedback,
            f = feedback[t-1]
            c = model_pred[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]


def rw_symmetric_LR(x, *args):

    """
    Tracks confidence and updates the PE with this confidence prediction.
    Calculates the sum of negative log likelihood (NLL) for a sequence of
    trials using the Rescorla-Wagner model with symmetric learning rates.
    The model updates confidence levels based on prediction errors (PE),
    which are derived from the difference between feedback and prior
    confidence.

    OBS! This model needs the first confidence value to be from a baseline
    trial. This mean n_trials shoould also be 1 trial longer than usual and
    the first feedback value could be any number as feedback[0] is never
    used (as it is assumed to be a baseline trial).

    Parameters:
    x (tuple): Contains alpha (learning rate) and sigma (standard deviation
               for the likelihood calculations).
    args (tuple): Includes arrays for confidence levels, feedback, and the
                  total number of trials.

    Returns:
    float: The total NLL for all trials, indicating the model's fit to
           the observed data.

    The function processes each trial, updating confidence levels based
    on feedback. It ensures updated confidence levels remain within 0-100.
    NLLs are calculated using a truncated normal distribution, adjusting
    for the bounded nature of confidence levels. The function ignores the
    baseline trial in its final output.
    """

    alpha, sigma, bias = x
    confidence, feedback, n_trials, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = bias

        if t > 0:

            # Get previous confidence estimate and feedback
            f = feedback[t-1]
            c = model_pred[t-1]

            # Update rule
            c_pred = c + alpha*(f - c)

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Calculate probabilities and Negative log likelihood (NLL)
    probabilities, y_values = calc_norm_prob_vectorized(confidence,
                                                        model_pred,
                                                        sigma_vec)
    nlls = -np.log(probabilities + epsilon)

    return np.sum(nlls[trial_index])


def win_stay_lose_shift_trial(x, *args):

    """
    Computes the negative loglikelihood (NLL) for a series of trials
    using the 'win-stay, lose-shift' strategy. This model adjusts the
    confidence based on feedback, employing 'win-stay' if feedback is
    within a certain boundary of previous confidence, or 'lose-shift'
    otherwise. The likelihood for each confidence level is calculated
    based on a normal distribution centered on prior confidence for
    wins, or its inverse for losses.

    Parameters:
    x (tuple): Contains sigma (standard deviation) and win_boundary
               (boundary for win condition).
    args (tuple): Includes arrays of confidence levels, feedback, and
                  total number of trials.

    Returns:
    list: Contains the NLL vector, probability distributions, and
          adjusted confidence levels for each trial.

    Note:
    The model assumes a prediction range of 100 and corrects for
    entry errors beyond this range. Uses a small epsilon to prevent
    log of zero. The likelihood is determined by normal or inverted
    distributions based on win/lose conditions.
    """

    sigma, win_boundary = x
    confidence, feedback, n_trials = args

    options = np.linspace(0, 100, 100+1)
    nll_vector = np.zeros(n_trials)
    prob_dists = []

    for t in range(n_trials):

        epsilon = 1e-10
        if t == 0:  # first trial, only included to index previous c and f.
            continue
        else:
            # Get previous confidence value and set
            f = feedback[t-1]
            c = int(confidence[t-1])

            # Set win boundary
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary

            if f > lower_bound and f < upper_bound:  # win-trial

                # Calculate probability of prediction c (p_pred)
                # and the probabilities of all options  (p_dist)
                p_pred, p_dist = calc_norm_prob_vectorized(
                                                    np.array([confidence[t]]),
                                                    np.array([c]),
                                                    np.array([sigma]))

                prob_dists.append(p_dist)

            else:  # lose-trial

                # Calculate probabilities and Negative log likelihood (NLL)
                p_pred, p_dist = calc_inverted_norm_prob_vectorized(
                                                    np.array([confidence[t]]),
                                                    np.array([c]),
                                                    np.array([sigma]))

                prob_dists.append(p_dist)

            # Get negative log likelihood of reported confidence
            nll_vector[t] = -np.log(p_dist[0][int(confidence[t])] + epsilon)

    # Remove first trial
    nll_vector = nll_vector[1:]
    confidence = confidence[1:]

    return [nll_vector, prob_dists, confidence]


def win_stay_lose_shift(x, *args):

    """
    Calculates the total negative loglikelihood (NLL) for a sequence
    of trials using a 'win-stay, lose-shift' strategy. This model adjusts
    confidence levels based on feedback within a specified boundary.
    The 'win-stay' strategy is employed when feedback is within this
    boundary; otherwise, 'lose-shift' is applied. The likelihood is
    calculated using a normal distribution centered on prior confidence
    for wins or its inverse for losses.

    Parameters:
    x (tuple): Contains sigma (standard deviation) and win_boundary
               (boundary for win condition).
    args (tuple): Includes arrays of confidence levels, feedback, and
                  total number of trials.

    Returns:
    float: The total NLL for all trials.

    Note:
    The model assumes a prediction range up to 100. Adds a small epsilon to
    prevent log of zero. The likelihood calculation depends on the win/lose
    condition for each trial.
    """

    sigma, win_boundary = x
    confidence, feedback, n_trials, trial_index = args

    options = np.linspace(0, 100, 101)
    nll_vector = np.zeros(n_trials)
    epsilon = 1e-10
    for t in range(n_trials):


        if t == 0:  # first trial, only included to index previous c and f.
            continue
        else:
            # Get previous confidence value and set
            f = feedback[t-1]
            c = int(confidence[t-1])

            # Set win boundary
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary

            if f > lower_bound and f < upper_bound:  # win-trial

                # Calculate probability of prediction c (p_pred)
                # and the probabilities of all options  (p_dist)
                p_pred, p_dist = calc_norm_prob_vectorized(
                                                    np.array([confidence[t]]),
                                                    np.array([c]),
                                                    np.array([sigma]))

            else:  # lose-trial

                # Calculate probabilities and Negative log likelihood (NLL)
                p_pred, p_dist = calc_inverted_norm_prob_vectorized(
                                                    np.array([confidence[t]]),
                                                    np.array([c]),
                                                    np.array([sigma]))

            # Get negative log likelihood of reported confidence
            nll_vector[t] = -np.log(p_dist[0][int(confidence[t])] + epsilon)

    # Remove first trial in session
    nll_vector = nll_vector[1:]

    return np.sum(nll_vector[trial_index])


def random_model_w_bias_trial(x, *args):
    """
    Calculates the total negative loglikelihood (NLL) for a series of
    trials using a biased random choice model. This model generates a
    probability distribution biased towards certain options, defined by
    a normal distribution centered around a specified mean. It evaluates
    the likelihood of observed confidence scores based on this biased
    distribution.

    Parameters:
    x (tuple): Contains mean_option (the mean of the biasing distribution)
               and sigma (standard deviation of the distribution).
    args (tuple): Includes an array of confidence scores and the number of
                  trials.

    Returns:
    list: Contains the NLL vector, probability distributions for each trial,
          and adjusted confidence scores.

    Note:
    The model assumes a prediction range up to 100. Incorporates a small
    epsilon to prevent log of zero errors. Each trial's likelihood is
    calculated based on the biased probability distribution.The baseline is
    ignored in the output.
    """

    mean_option, sigma = x
    confidence, n_trials = args

    options = np.linspace(0, 100, 101)
    nll_vector = np.zeros(n_trials)
    prob_dists = []
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:  # First trial, nll not included for this trial

            continue

        else:

            # Calculate probabilities and Negative log likelihood (NLL)
            p_pred, p_dist = calc_inverted_norm_prob_vectorized(
                                                        np.array([confidence[t]]),
                                                        np.array([mean_option]),
                                                        np.array([sigma]))

            prob_dists.append(p_dist)

            # Get negative log likelihood of choosing that confidence
            nll_vector[t] = -np.log(p_dist[0][int(confidence[t])] + epsilon)

    # remove baseline
    nll_vector = nll_vector[1:]
    confidence = confidence[1:]

    return [nll_vector, prob_dists, confidence]


def random_model_w_bias(x, *args):

    """
    Calculates the NLL of a biased random choice model.

    This function computes the NLL for choices based on a probability
    distribution biased towards certain options. The bias is characterized
    by a normal distribution centered around a mean option with a specified
    standard deviation. It assesses the likelihood of observed choices
    (confidence scores) according to this biased distribution.

    Parameters:
    x (tuple): Contains two elements:
               - mean_option (float): The mean of the biasing normal
                                      distribution.
               - sigma (float): The standard deviation of the distribution.
    args (tuple): Additional arguments:
                  - confidence (array-like): Array of confidence scores.
                  - n_trials (int): Number of trials in the dataset.

    Returns:
    float: Total NLL for the given set of confidence scores.

    Note:
    Assumes a fixed prediction range of 100. A small epsilon is added to
    prevent log of zero calculations. The baseline is ignored in the output.
    """

    mean_option, sigma = x
    confidence, n_trials, trial_index = args

    options = np.linspace(0, 100, 101)
    nll_vector = np.zeros(n_trials)
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:  # first trial, only included to index passed c and f.
            continue
        else:

            # Calculate probabilities and Negative log likelihood (NLL)
            p_pred, p_dist = calc_norm_prob_vectorized(
                                                        np.array([confidence[t]]),
                                                        np.array([mean_option]),
                                                        np.array([sigma]))

            # Get negative log likelihood of choosing that confidence
            nll_vector[t] = -np.log(p_dist[0][int(confidence[t])] + epsilon)

    nll_vector = nll_vector[1:]

    return np.sum(nll_vector[trial_index])

def fit_random_model(prediction_range, n_trials):

    """
    Calculate total negative log likelihood for random choices over trials.

    This function simulates making predictions randomly and uniformly over a
    specified range, calculating the negative log likelihood (NLL) for each
    trial. The total NLL is computed by summing the NLLs of all trials, based
    on the assumption that each choice within the prediction range is equally
    likely.

    Parameters:
    prediction_range (int): Range of possible prediction values, determining
                            the probability of each random choice.
    n_trials (int): Number of trials for which to calculate and sum the NLL.

    Returns:
    float: Cumulative NLL value for all trials, representing the penalty for
           random predictions over the given number of trials and prediction
           range. One baseline trial is expected and therefore ignored.

    Example:
    >>> fit_random_model(10, 5)
    11.512925464970229
    """
    # Remove first trial of session to make output comparable
    n_trials = n_trials-1

    random_choice_probability = 1/prediction_range
    log_likelihood = math.log(random_choice_probability)
    nll_vector = [-log_likelihood]*n_trials

    return np.sum(nll_vector)

def random_model(x, *args):

    """
    Calculate total negative log likelihood for random choices over trials.

    This function simulates making predictions randomly and uniformly over a
    specified range, calculating the negative log likelihood (NLL) for each
    trial. The total NLL is computed by summing the NLLs of all trials, based
    on the assumption that each choice within the prediction range is equally
    likely.

    Parameters:
    prediction_range (int): Range of possible prediction values, determining
                            the probability of each random choice.
    n_trials (int): Number of trials for which to calculate and sum the NLL.

    Returns:
    float: Cumulative NLL value for all trials, representing the penalty for
           random predictions over the given number of trials and prediction
           range. One baseline trial is expected and therefore ignored.

    Example:
    >>> fit_random_model(10, 5)
    11.512925464970229
    """

    # Variables
    prediction_range, n_trials, trial_index = args
    # nll vector
    nll_vector = np.zeros(n_trials)
    # get negative log likelihood given random choice
    random_choice_probability = 1/prediction_range
    log_likelihood = math.log(random_choice_probability)
    nll_vector = np.array([-log_likelihood]*n_trials)

    # Remove first trial for fair comparison
    nll_vector = nll_vector[1:]

    return np.sum(nll_vector[trial_index])

#-----------------------------------------------------------------------------
# Model fitting functions
#-----------------------------------------------------------------------------

# =============================================================================
# def normalize(params, bounds):
#     return [(p - b[0]) / (b[1] - b[0]) for p, b in zip(params, bounds)]
#
# def inverse_normalize(norm_params, bounds):
#     return [p * (b[1] - b[0]) + b[0] for p, b in zip(norm_params, bounds)]
# =============================================================================

def normalize(params, bounds):
    return [np.clip((p - b[0]) / (b[1] - b[0]), 0, 1)
            for p, b in zip(params, bounds)]

def inverse_normalize(norm_params, bounds):
    return [np.clip(p * (b[1] - b[0]) + b[0], b[0], b[1])
            for p, b in zip(norm_params, bounds)]

def fit_model(model, args, bounds, n_trials, start_value_number=10,
              solver="L-BFGS-B", bias_model_best_params=False):

    # Normalize bounds for the optimization process
    norm_bounds = [(0, 1) for _ in bounds]

    # Wrap model function to inverse normalize parameters before evaluation
    def model_wrapper(norm_params, *args):
        params = inverse_normalize(norm_params, bounds)
        return model(params, *args) # Returns model function

    # Generate normalized start values
    start_ranges = []
    for bound in norm_bounds:
        lower_bound, upper_bound = bound
        start_range = np.random.uniform(lower_bound, upper_bound,
                                        size=start_value_number)
        start_ranges.append(start_range)

    # Optimization process
    result_list = []
    nll_list = []
    for i, norm_start_values in enumerate(zip(*start_ranges)):

        # Use best bias params for first run
        if i == 0 and bias_model_best_params!=False:
            norm_start_values = bias_model_best_params

        # Fit model to data
        result = minimize(model_wrapper, norm_start_values, args=args,
                          bounds=norm_bounds, method=solver)

        # Inverse normalize optimized parameters
        # to get them back in their original scale
        best_params = inverse_normalize(result.x, bounds)
        nll = result.fun

        # McFadden pseudo r2
        n_options = 101 # levels of confidence (i.e., 0,1,2... ...99, 100)
        nll_model = nll
        nll_null = -n_trials * np.log(1 / n_options)
        pseudo_r2 = 1 - (nll_model / nll_null)

        # AIC and BIC
        k = len(best_params)
        aic = 2*k + 2*nll
        bic = k*np.log(n_trials) + 2*nll

        # Append to result and nll lists
        result_list.append(best_params + [nll, aic, bic, pseudo_r2])
        nll_list.append(nll)

    # Index best result
    best_result_index = nll_list.index(min(nll_list))
    best_result = result_list[best_result_index]

    return best_result

def fit_model_with_LOO_cv(model, args, bounds, n_trials,
                      start_value_number=50,
                      solver="L-BFGS-B",
                      bias_model_best_params=False):

    # Normalize bounds for the optimization process
    norm_bounds = [(0, 1) for _ in bounds]

    # Leave-One-Out Cross-Validation setup
    loo = LeaveOneOut()

    # Placeholder for cross-validation results
    cv_results = []

    n_adjust = 1
    for train_index, test_index in loo.split(np.arange(n_trials-n_adjust)):

        # Wrap model function to inverse normalize parameters before evaluation
        def model_wrapper(norm_params, *args):
            params = inverse_normalize(norm_params, bounds)
            return model(params, *args) # Returns model function

        # Generate normalized start values
        start_ranges = []
        for bound in norm_bounds:
            lower_bound, upper_bound = bound
            start_range = np.random.uniform(lower_bound, upper_bound,
                                            size=start_value_number)
            start_ranges.append(start_range)

        # Optimization process
        result_list = []
        nll_list = []
        for i, norm_start_values in enumerate(zip(*start_ranges)):
            # Use best bias params for first run
            if i == 0 and bias_model_best_params != False:
                norm_start_values = bias_model_best_params

            # Fit model to data
            args_train = args + (train_index,) # Add index for trails used for training
            result = minimize(model_wrapper, norm_start_values,
                              args=args_train, bounds=norm_bounds,
                              method=solver)

            # Inverse normalize optimized parameters
            # to get them back in their original scale
            best_params = inverse_normalize(result.x, bounds)
            nll = result.fun

            # Evaluate on validation data
            args_test = args + (test_index,) # Add index for trial used for test
            val_nll = model(best_params, *args_test)
            #val_nll = model_wrapper(result.x, args_train)

            # McFadden pseudo r2
            n_options = 101 # levels of confidence (i.e., 0,1,2... ...99,100)
            nll_model = nll
            nll_null = -n_trials * np.log(1 / n_options)
            pseudo_r2 = 1 - (nll_model / nll_null)

            # AIC and BIC
            # OBS! As these are calculated using cross-validated NLL,
            # the AIC and BIC now only reflect a model complexity penalty after
            # overfitting have already been controlled for by the
            # cross-validation. Hence, the AIC and BIC measures should not be
            # thought of as a control for overfitting in this case.

            k = len(best_params)
            if model==random_model:
                k=0
               # print('random model', 'k =', k)
            else:
               # print(f'{model}', 'k =', k)
                pass
            aic = 2*k + 2*val_nll
            bic = k*np.log(n_trials) + 2*val_nll

            # Append to result and nll lists
            result_list.append(best_params + [val_nll, aic, bic, pseudo_r2])
            nll_list.append(val_nll)

        # Index best result
        best_result_index = nll_list.index(min(nll_list))
        best_result = result_list[best_result_index]

        # Append the best result for this fold
        cv_results.append(best_result)

    # Average results over all folds
    averaged_result = np.mean(cv_results, axis=0)

    return averaged_result

def fit_model_with_cv(model, args, bounds, n_trials,
                      start_value_number=50,
                      solver="L-BFGS-B",
                      bias_model_best_params=False):

    # Normalize bounds for the optimization process
    norm_bounds = [(0, 1) for _ in bounds]

    # Placeholder for cross-validation results
    cv_results = []

    # Remove 2 trials:
    # The first trial in session is not fit
    # And another trial needs to be removed for the cross-validation.
    n_adjust_sess = 1 # 1 trial removed from session
    n_adjust_cv = 1 # 1 trial removed from cv splits
    n_trials_adjusted = n_trials-n_adjust_sess
    n_splits_cv = n_trials_adjusted-n_adjust_cv # Train on past, test on future.

    # Timeseries Cross-Validation setup
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=n_splits_cv)

    for train_index, test_index in tscv.split(np.arange(n_trials_adjusted)):

        # Wrap model function to inverse normalize parameters before evaluation
        def model_wrapper(norm_params, *args):
            params = inverse_normalize(norm_params, bounds)
            return model(params, *args) # Returns model function

        # Generate normalized start values
        start_ranges = []
        for bound in norm_bounds:
            lower_bound, upper_bound = bound
            start_range = np.random.uniform(lower_bound, upper_bound,
                                            size=start_value_number)
            start_ranges.append(start_range)

        # Optimization process
        result_list = []
        nll_list = []
        for i, norm_start_values in enumerate(zip(*start_ranges)):
            # Use best bias params for first run
            if i == 0 and bias_model_best_params != False:
                norm_start_values = bias_model_best_params

            # Fit model to data
            args_train = args + (train_index,) # Add index for trails used for training
            result = minimize(model_wrapper, norm_start_values,
                              args=args_train, bounds=norm_bounds,
                              method=solver)

            # Inverse normalize optimized parameters
            # to get them back in their original scale
            best_params = inverse_normalize(result.x, bounds)
            nll = result.fun

            # Evaluate on validation data
            args_test = args + (test_index,) # Add index for trial used for test
            val_nll = model(best_params, *args_test)
            #val_nll = model_wrapper(result.x, args_train)

            # McFadden pseudo r2
            n_options = 101 # levels of confidence (i.e., 0,1,2... ...99,100)
            nll_model = nll
            nll_null = -n_trials * np.log(1 / n_options)
            pseudo_r2 = 1 - (nll_model / nll_null)

            # AIC and BIC
            # OBS! As these are calculated using cross-validated NLL,
            # the AIC and BIC now only reflect a model complexity penalty after
            # overfitting have already been controlled for by the
            # cross-validation. Hence, the AIC and BIC measures should not be
            # thought of as a control for overfitting in this case.

            k = len(best_params)
            if model==random_model:
                k=0
               # print('random model', 'k =', k)
            else:
               # print(f'{model}', 'k =', k)
                pass
            aic = 2*k + 2*val_nll
            bic = k*np.log(n_trials) + 2*val_nll

            # Append to result and nll lists
            result_list.append(best_params + [val_nll, aic, bic, pseudo_r2])
            nll_list.append(val_nll)

        # Index best result
        best_result_index = nll_list.index(min(nll_list))
        best_result = result_list[best_result_index]

        # Append the best result for this fold
        cv_results.append(best_result)

    # Average results over all folds
    averaged_result = np.mean(cv_results, axis=0)

    return averaged_result

# Function to process data for a single session
def process_session(df_s):

    # Participant and session and model
    participant = df_s.pid.unique()[0]
    session = df_s.session.unique()[0]
    model = df_s.model.unique()[0]
    pid_session_model_tuple = (participant, session, model)

    # Remove baseline
    df_s = df_s[df_s.condition != 'baseline']

    # Calculate absolute trial error as the average across subtrials
    df_s['difference'] = abs(df_s['estimate'] - df_s['correct'])
    abs_error_avg = df_s.groupby('trial')['difference'].mean()

    # Only keep first row of every subtrial (one row = one trial)
    df_s = df_s.drop_duplicates(subset='trial', keep='first')

    # Condition
    condition = df_s.condition.unique()[0]

    # N trials
    n_trials = len(df_s)

    # Calculate trial-by-trial metrics
    confidence = df_s.confidence_sim.values # Simulated estimates.
    # For dot-estimate performance and feedback, we could use random noise, but
    # using participant performance and feedback has the advantage that it
    # tells us if we can recover models and parameters from the performance
    # and feedback values we know are in the data.
    feedback = df_s.feedback.values # Feedback from participant.
    performance = -abs_error_avg.values # Performance from participant.

    # Random confidence model
    bounds = [(0, 0)]
    results = fit_model_with_cv(model=random_model,
                                args=(101, n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")

    nll_random = results[1].item()
    aic_random = results[2].item()
    bic_random = results[3].item()
    pseudo_r2_random = results[4].item()

    # Biased confidence model
    mean_bound = (0, 100)
    sigma_bound = (1, 20)
    bounds = [(mean_bound[0], mean_bound[1]),
              (sigma_bound[0], sigma_bound[1])]
    results = fit_model_with_cv(model=random_model_w_bias,
                                args=(confidence, n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")
    mean_bias = results[0].item()
    sd_bias= results[1].item()
    nll_bias = results[2].item()
    aic_bias = results[3].item()
    bic_bias = results[4].item()
    pseudo_r2_bias = results[5].item()

    # Win-stay-lose-shift model
    sigma_bound = (1, 20)
    win_bound = (1, 100)
    bounds = [(sigma_bound[0], sigma_bound[1]),
              (win_bound[0], win_bound[1])]
    results_win_stay = fit_model_with_cv(model=win_stay_lose_shift,
                                         args=(confidence,
                                               feedback,
                                               n_trials),
                                         bounds=bounds,
                                         n_trials=n_trials,
                                         start_value_number=50,
                                         solver="L-BFGS-B")

    std_WSLS_array = results_win_stay[0].item()
    win_boundary_WSLS_array = results_win_stay[1].item()
    nll_win_stay = results_win_stay[2].item()
    aic_win_stay = results_win_stay[3].item()
    bic_win_stay = results_win_stay[4].item()
    pseudo_r2_win_stay = results_win_stay[5].item()

    # Rescorla-Wagner model
    alpha_bound = (0, 1)
    sigma_bound = (1, 20)
    bias_bound = (0, 100)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]
    results_rw_symm = fit_model_with_cv(model=rw_symmetric_LR,
                                        args=(confidence, feedback, n_trials),
                                        bounds=bounds, n_trials=n_trials,
                                        start_value_number=50,
                                        solver="L-BFGS-B")
    alpha_rw_symm = results_rw_symm[0].item()
    sigma_rw_symm = results_rw_symm[1].item()
    bias_rw_symm = results_rw_symm[2].item()
    nll_rw_symm = results_rw_symm[3].item()
    aic_rw_symm = results_rw_symm[4].item()
    bic_rw_symm = results_rw_symm[5].item()
    pseudo_r2_rw_symm = results_rw_symm[6].item()

    # Rescorla wagner Condition alpha model
    alpha_neut_bound = (0, 1)  # Alpha neut
    alpha_pos_bound =  (0, 1)  # Alpha pos
    alpha_neg_bound =  (0, 1)  # Alpha neg
    sigma_bound = (1, 20)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_neut_bound[0], alpha_neut_bound[1]),
              (alpha_pos_bound[0], alpha_pos_bound[1]),
              (alpha_neg_bound[0], alpha_neg_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]
    results_rw_cond = fit_model_with_cv(model=rw_cond_LR,
                                args=(confidence, feedback, n_trials,
                                      df_s.condition.values),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")
    alpha_neut_rw_cond = results_rw_cond[0].item()
    alpha_pos_rw_cond = results_rw_cond[1].item()
    alpha_neg_rw_cond = results_rw_cond[2].item()
    sigma_rw_cond = results_rw_cond[3].item()
    bias_rw_cond = results_rw_cond[4].item()
    nll_rw_cond = results_rw_cond[5].item()
    aic_rw_cond = results_rw_cond[6].item()
    bic_rw_cond = results_rw_cond[7].item()
    pseudo_r2_rw_cond = results_rw_cond[8].item()

    # Choice Kernel model
    alpha_bound = (0, 1)
    sigma_bound = (1, 20)
    bias_bound = (0, 100)
    beta_bound = (40, 200)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1])]
    results_ck = fit_model_with_cv(model=choice_kernel,
                                   args=(confidence, n_trials),
                                   bounds=bounds,
                                   n_trials=n_trials,
                                   start_value_number=50,
                                   solver="L-BFGS-B")
    alpha_ck = results_ck[0].item()
    sigma_ck = results_ck[1].item()
    bias_ck = results_ck[2].item()
    beta_ck = results_ck[3].item()
    nll_ck = results_ck[4].item()
    aic_ck = results_ck[5].item()
    bic_ck = results_ck[6].item()
    pseudo_r2_ck = results_ck[7].item()

    # RW + Choice Kernel model
    alpha_bound = (0, 1)
    alpha_ck_bound = (0, 1)
    sigma_bound = (1, 20)
    sigma_ck_bound = (1, 20)
    bias_bound = (0, 100)
    beta_bound = (40, 200)
    beta_ck_bound = (40, 200)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (alpha_ck_bound[0], alpha_ck_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (sigma_ck_bound[0], sigma_ck_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1]),
              (beta_ck_bound[0], beta_ck_bound[1])]
    results_rwck = fit_model_with_cv(model=RW_choice_kernel,
                                     args=(feedback, confidence, n_trials),
                                     bounds=bounds,
                                     n_trials=n_trials,
                                     start_value_number=50,
                                     solver="L-BFGS-B")
    alpha_rwck = results_rwck[0].item()
    alpha_ck_rwck = results_rwck[1].item()
    sigma_rwck = results_rwck[2].item()
    sigma_ck_rwck = results_rwck[3].item()
    bias_rwck = results_rwck[4].item()
    beta_rwck = results_rwck[5].item()
    ck_beta_rwck = results_rwck[6].item()
    nll_rwck = results_rwck[7].item()
    aic_rwck = results_rwck[8].item()
    bic_rwck = results_rwck[9].item()
    pseudo_r2_rwck = results_rwck[10].item()

    # Rescorla-Wagner Performance Delta model (RWPD)
    alpha_bound = (0, 1)
    sigma_bound = (1, 20)
    bias_bound = (0, 100)
    w_rw_bound = (0, 5)
    w_delta_p_bound = (2, 8)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (w_rw_bound[0], w_rw_bound[1]),
              (w_delta_p_bound[0], w_delta_p_bound[1])]
    results_rwpd = fit_model_with_cv(model=rwpd,
                                           args=(confidence,
                                                 feedback,
                                                 n_trials,
                                                 performance),
                                           bounds=bounds,
                                           n_trials=n_trials,
                                           start_value_number=50,
                                           solver="L-BFGS-B")
    alpha_rwpd = results_rwpd[0].item()
    sigma_rwpd = results_rwpd[1].item()
    bias_rwpd = results_rwpd[2].item()
    w_rw_rwpd = results_rwpd[3].item()
    w_performance_rwpd = results_rwpd[4].item()
    nll_rwpd = results_rwpd[5].item()
    aic_rwpd = results_rwpd[6].item()
    bic_rwpd = results_rwpd[7].item()
    pseudo_r2_rwpd = results_rwpd[8].item()

    # Store session metrics
    # The p at the end of the key name stands for person as each value is
    # personal.
    session_metrics = {
        'pid': participant,
        'session': session,
        'model': model,
        'condition': condition,
        'n_trials': n_trials,
        'nll_random': nll_random,
        'aic_random': aic_random,
        'bic_random': bic_random,
        'pseudo_r2_random': pseudo_r2_random,
        'mean_bias': mean_bias,
        'sd_bias': sd_bias,
        'nll_bias': nll_bias,
        'aic_bias': aic_bias,
        'bic_bias': bic_bias,
        'pseudo_r2_bias': pseudo_r2_bias,
        'std_WSLS': std_WSLS_array,
        'win_boundary_WSLS': win_boundary_WSLS_array,
        'nll_win_stay': nll_win_stay,
        'aic_win_stay': aic_win_stay,
        'bic_win_stay': bic_win_stay,
        'pseudo_r2_win_stay': pseudo_r2_win_stay,
        'alpha_rw_symm': alpha_rw_symm,
        'sigma_rw_symm': sigma_rw_symm,
        'bias_rw_symm': bias_rw_symm,
        'nll_rw_symm': nll_rw_symm,
        'aic_rw_symm': aic_rw_symm,
        'bic_rw_symm': bic_rw_symm,
        'pseudo_r2_rw_symm': pseudo_r2_rw_symm,
        'alpha_neut_rw_cond': alpha_neut_rw_cond,
        'alphaos_rw_cond': alpha_pos_rw_cond,
        'alpha_neg_rw_cond': alpha_neg_rw_cond,
        'sigma_rw_cond': sigma_rw_cond,
        'bias_rw_cond': bias_rw_cond,
        'nll_rw_cond': nll_rw_cond,
        'aic_rw_cond': aic_rw_cond,
        'bic_rw_cond': bic_rw_cond,
        'pseudo_r2_rw_cond': pseudo_r2_rw_cond,
        'alpha_ck': alpha_ck,
        'sigma_ck': sigma_ck,
        'bias_ck': bias_ck,
        'beta_ck': beta_ck,
        'nll_ck': nll_ck,
        'aic_ck': aic_ck,
        'bic_ck': bic_ck,
        'pseudo_r2_ck': pseudo_r2_ck,
        'alpha_rwck': alpha_rwck,
        'alpha_ck_rwck': alpha_ck_rwck,
        'sigma_rwck': sigma_rwck,
        'sigma_ck_rwck': sigma_ck_rwck,
        'bias_rwck': bias_rwck,
        'beta_rwck': beta_rwck,
        'ck_beta_rwck': ck_beta_rwck,
        'nll_rwck': nll_rwck,
        'aic_rwck': aic_rwck,
        'bic_rwck': bic_rwck,
        'pseudo_r2_rwck': pseudo_r2_rwck,
        'alpha_rwpd': alpha_rwpd,
        'sigma_rwpd': sigma_rwpd,
        'bias_rwpd': bias_rwpd,
        'w_rw_rwpd': w_rw_rwpd,
        'w_performance_rwpd': w_performance_rwpd,
        'nll_rwpd': nll_rwpd,
        'aic_rwpd': aic_rwpd,
        'bic_rwpd': bic_rwpd,
        'pseudo_r2_rwpd': pseudo_r2_rwpd,
    }

    return pid_session_model_tuple, session_metrics

#-----------------------------------------------------------------------------
# Main function to perform model fitting on EXP2 data
#-----------------------------------------------------------------------------

def main(df):

    # Unique pid, model, and sessions pairs
    unique_pairs = df[['pid', 'session', 'model']].drop_duplicates()

    # Convert the DataFrame to a list of tuples
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))
    # unique_pairs_list = [unique_pairs_list[0]]

    # Create a list of DataFrames, one for each unique pid-model-session combo
    df_list = [df[(df['pid'] == pid) &
                  (df['session'] == session) &
                  (df['model'] == model)].copy()
               for pid, session, model in unique_pairs_list]

    # Total number of unique combos
    num_unique_combos = len(unique_pairs_list)
    print(f"Number of pid-model-session combos: {num_unique_combos}")

    # Initialize result dictionaries for all participants
    results = {
        'pid': np.empty(num_unique_combos, dtype=object),  # Use dtype=object for strings
        'session': np.zeros(num_unique_combos),
        'model': np.zeros(num_unique_combos, dtype=object),
        'condition':  np.empty(num_unique_combos, dtype=object),  # Use dtype=object for strings
        'n_trials': np.zeros(num_unique_combos),
        'nll_random': np.zeros(num_unique_combos),
        'aic_random': np.zeros(num_unique_combos),
        'bic_random': np.zeros(num_unique_combos),
        'pseudo_r2_random': np.zeros(num_unique_combos),
        'mean_bias': np.zeros(num_unique_combos),
        'sd_bias': np.zeros(num_unique_combos),
        'nll_bias': np.zeros(num_unique_combos),
        'aic_bias': np.zeros(num_unique_combos),
        'bic_bias': np.zeros(num_unique_combos),
        'pseudo_r2_bias': np.zeros(num_unique_combos),
        'std_WSLS': np.zeros(num_unique_combos),
        'win_boundary_WSLS': np.zeros(num_unique_combos),
        'nll_win_stay': np.zeros(num_unique_combos),
        'aic_win_stay': np.zeros(num_unique_combos),
        'bic_win_stay': np.zeros(num_unique_combos),
        'pseudo_r2_win_stay': np.zeros(num_unique_combos),
        'alpha_rw_symm': np.zeros(num_unique_combos),
        'sigma_rw_symm': np.zeros(num_unique_combos),
        'bias_rw_symm': np.zeros(num_unique_combos),
        'nll_rw_symm': np.zeros(num_unique_combos),
        'aic_rw_symm': np.zeros(num_unique_combos),
        'bic_rw_symm': np.zeros(num_unique_combos),
        'pseudo_r2_rw_symm': np.zeros(num_unique_combos),
        'alpha_neut_rw_cond': np.zeros(num_unique_combos),
        'alphaos_rw_cond': np.zeros(num_unique_combos),
        'alpha_neg_rw_cond': np.zeros(num_unique_combos),
        'sigma_rw_cond': np.zeros(num_unique_combos),
        'bias_rw_cond': np.zeros(num_unique_combos),
        'nll_rw_cond': np.zeros(num_unique_combos),
        'aic_rw_cond': np.zeros(num_unique_combos),
        'bic_rw_cond': np.zeros(num_unique_combos),
        'pseudo_r2_rw_cond': np.zeros(num_unique_combos),
        'alpha_ck': np.zeros(num_unique_combos),
        'sigma_ck': np.zeros(num_unique_combos),
        'bias_ck': np.zeros(num_unique_combos),
        'beta_ck': np.zeros(num_unique_combos),
        'nll_ck': np.zeros(num_unique_combos),
        'aic_ck': np.zeros(num_unique_combos),
        'bic_ck': np.zeros(num_unique_combos),
        'pseudo_r2_ck': np.zeros(num_unique_combos),
        'alpha_rwck': np.zeros(num_unique_combos),
        'alpha_ck_rwck': np.zeros(num_unique_combos),
        'sigma_rwck': np.zeros(num_unique_combos),
        'sigma_ck_rwck': np.zeros(num_unique_combos),
        'bias_rwck': np.zeros(num_unique_combos),
        'beta_rwck': np.zeros(num_unique_combos),
        'ck_beta_rwck': np.zeros(num_unique_combos),
        'nll_rwck': np.zeros(num_unique_combos),
        'aic_rwck': np.zeros(num_unique_combos),
        'bic_rwck': np.zeros(num_unique_combos),
        'pseudo_r2_rwck': np.zeros(num_unique_combos),
        'alpha_rwpd': np.zeros(num_unique_combos),
        'sigma_rwpd': np.zeros(num_unique_combos),
        'bias_rwpd': np.zeros(num_unique_combos),
        'w_rw_rwpd': np.zeros(num_unique_combos),
        'w_performance_rwpd': np.zeros(num_unique_combos),
        'nll_rwpd': np.zeros(num_unique_combos),
        'aic_rwpd': np.zeros(num_unique_combos),
        'bic_rwpd': np.zeros(num_unique_combos),
        'pseudo_r2_rwpd': np.zeros(num_unique_combos),
    }

    # Use multiprocessing to process each participant in parallel
    with Pool(48) as pool:
        for sim_id, metrics in tqdm(pool.map(process_session, df_list),
                                        total=len(df_list)):

            # Get the index of the current session
            idx = unique_pairs_list.index(sim_id)

            # Process each key in the metrics dictionary
            for key, value in metrics.items():
                print(key, idx, value)
                if key in ['pid', 'model','condition']:  # String fields
                    results[key][idx] = value
                elif hasattr(value, 'item'):
                        results[key][idx] = value.item()  # Extract scalar if it's a NumPy or PyTorch object
                else:
                    results[key][idx] = value  # Use directly if it's already a Python scalar
              #  else:
              #      print(f"Unexpected type for key '{key}': {value} (type: {type(value)})")

    name = r'sim_EXP2_model_metrics_sessions_CV_super_deterministic_testing123.xlsx'

    df_m = pd.DataFrame(results)
    df_m.to_excel(name) # save to current directory

#-----------------------------------------------------------------------------
# Execution
#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Import data - Varied feedback condition (Experiment 2)
    data_file = r'simulated_conf_EXP2_data_more_deterministic.csv'
    #data_file = r'simulated_conf_EXP2_data_best_fit_params.csv'
    #data_file = r'simulated_conf_EXP2_data_super_deterministic.csv'
    full_path = data_file
    df = pd.read_csv(full_path, low_memory=False)

    #df =  df[df['pid']=='5c884b29c2ceec001719b1e4']

    # Run main
    main(df)


