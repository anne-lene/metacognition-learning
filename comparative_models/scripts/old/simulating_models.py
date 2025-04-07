# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:44:41 2024

@author: carll
"""

# Simulating models on paramters drawn randomly from best-fit parameter range

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
import matplotlib.pyplot as plt

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

def sim_norm_prob_vectorized(mean, sigma,
                             lower_bound=0, upper_bound=100):
    """
     Generate confidence estimates by sampling from a normal
     distribution truncated between lower_bound and upper_bound.
     Returns both the sampled values and the probability
     distributions used for sampling.
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

    # Simulate confidence values by sampling based on the probabilities in y_values
    simulated_confidence = np.array([
        np.random.choice(x_values, p=prob_dist) for prob_dist in y_values
    ])

    return simulated_confidence, y_values

def sim_inverted_norm_prob_vectorized(mean, sigma,
                                       lower_bound=0, upper_bound=100):
    """Same function as above, but now returns the
    inverted and normalized y_values and simulated confidence"""

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

    # Simulate confidence values by sampling based on the probabilities in y_values
    simulated_confidence = np.array([
        np.random.choice(x_values, p=prob_dist) for prob_dist in y_values
    ])

    return simulated_confidence, y_values

#----------------------------------------------------------------------------
# Models
#----------------------------------------------------------------------------
def RWP_sim (x, *args):

    """
    Sim RWP model returning simulated confidence estimates
    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args
    """

    alpha, sigma, w_RW, w_P = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial
            model_pred[t] = confidence[t]

            # Calculate probabilities and simulate confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_pred[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]

        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = model_pred[t-1] #c_rw[t-1]

            # Update rule
            c_rw[t]  = c + alpha*(f - c)

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance influence on confidence
            # The +100 establish a threshold for when the error size should be
            # considered so large that it can be ignored.
            c_P = performance[t] + 100

            model_pred[t] = max(0, min(100, (w_RW*c_rw[t]) + (w_P*c_P)))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]

    return conf_vec, p_options

def RWP_trial (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the change in performance
    since the last trial (delta P).

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

            # Get performance influence on confidence
            # The +100 establish a threshold for when the error size should be
            # considered so large that it can be ignored.
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
    Confidence is updated as the weighted sum of: (1)
    confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
    the averge performance (P) on the last 3 subtrials.
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

            # Get performance influence on confidence
            # The +100 establish a threshold for when the error size should be
            # considered so large that it can be ignored.
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

def RWPD_sim (x, *args):

    """
    Sim RWPD model retuning simulated confidence estimates
    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args
    """

    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial
            model_pred[t] = confidence[t]
            c_rw[t] = confidence[t]
            conf_vec[t] = confidence[t]
        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance delta
            delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between -100 and 100.
            delta_p = max(-100, min(100, delta_p))

            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_PD*delta_p))))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     lower_bound=0,
                                                     upper_bound=100)
            conf_vec[t] = conf_sim[0]

    return conf_vec

def RWPD_trial (x, *args):

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
            delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between -100 and 100.
            delta_p = max(-100, min(100, delta_p))

            # Get the prediction as the weighted sum
            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_PD*delta_p))))

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

def RWPD (x, *args):

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
            delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between -100 and 100.
            delta_p = max(-100, min(100, delta_p))

            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_PD*delta_p))))

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

def RWCK_sim(x, *args):

    """
     Sim RWCK model returning simulated confidence estimates.
     alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck = x
     feedback, confidence, n_trials = args
    """

    alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck = x
    feedback, confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    model_pred = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)

    for t in range(n_trials):

        if t == 0:

            # Get starting probabilities
            t1_probs = np.full(array_length, value)

            # Set initial confidence value
            t1_probs[int(confidence[t])] = 1

            # Smooth distribution
            t1_probs = gaussian_filter1d(t1_probs, sigma)

            # Normalize to get probabilities
            probabilities = t1_probs / np.sum(t1_probs)

            # Simulate confidence estimate
            conf_vec[t] = np.random.choice(range(0,101), p=probabilities)

        if t > 0:
            # Update "value" --------------------------------------------

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = conf_vec[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
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

            # Simulate confidence estimate
            conf_vec[t] = np.random.choice(range(0,101), p=probabilities)

    return conf_vec


def RWCK_trial(x, *args):

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


def RWCK(x, *args):

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


def CK_sim(x, *args):

    """
     Sim Choice kerel confidence estimates.
     alpha, sigma, beta = x
     confidence, n_trials = args
    """

    alpha, sigma, beta = x
    confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    conf_vec = np.zeros(n_trials)

    for t in range(n_trials):

        if t == 0:

            # Get starting probabilities
            t1_probs = np.full(array_length, value)

            # Make bias the most likely
            t1_probs[int(confidence[t])] = 1

            # Smooth distribution
            t1_probs = gaussian_filter1d(t1_probs, sigma)

            # Normalize to get probabilities
            probabilities = t1_probs / np.sum(t1_probs)

            # Simulate confidence estimate
            conf_vec[t] = np.random.choice(range(0,101), p=probabilities)

        if t > 0:

            # Get previous confidence value
            c = int(conf_vec[t-1])

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

            # Simulate confidence estimate
            conf_vec[t] = np.random.choice(range(0,101), p=probabilities)

    return conf_vec

def CK_trial(x, *args):
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


def CK(x, *args):

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


def RW_cond_sim(x, *args):

    """
    Sim rw_cond and return simulated confidence estiamtes.
    alpha_neut, alpha_pos, alpha_neg, sigma = x
    confidence, feedback, n_trials, condition = args
    """

    alpha_neut, alpha_pos, alpha_neg, sigma = x
    confidence, feedback, n_trials, condition = args

    model_pred = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = confidence[t]

            # Simulate probabilities and confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_pred[t]]),
                                                    np.array([sigma]))

            conf_vec[t] = conf_sim[0]

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

            # Simulate probabilities and confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_pred[t]]),
                                                    np.array([sigma]))

            conf_vec[t] = conf_sim[0]

    return conf_vec


def RW_cond_trial(x, *args):
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

def RW_cond(x, *args):

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


def RW_sim(x, *args):

    """
    Sim RW model and return simulated confidence.
    alpha, sigma = x
    confidence, feedback, n_trials = args
    """

    alpha, sigma = x
    confidence, feedback, n_trials = args

    model_mean = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_mean[t] = confidence[t]

            # Calculate probabilities and simulate confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_mean[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]

        if t > 0:

            # Get previous confidence estimate and feedback
            f = feedback[t-1]
            c = model_mean[t-1]

            # Update rule
            c_pred = c + alpha*(f - c)

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_mean[t] = c_pred

            # Calculate probabilities and simulate confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_mean[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]

    return conf_vec, y_values

def RW_trial(x, *args):
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


def RW(x, *args):

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


def WSLS_sim(x, *args):

    """
    Sim WSLS and return sim confidence.
    sigma, win_boundary = x
    confidence, feedback, n_trials = args
    """

    sigma, win_boundary = x
    confidence, feedback, n_trials = args
    conf_vec = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:  # First trial, only included to index previous c and f.

            f = feedback[t]
            c = confidence[t]

            conf_sim, p_dist = sim_norm_prob_vectorized(

                                               np.array([c]),
                                               np.array([sigma]))

            # Save simulated confidence of trial t
            conf_vec[t] = conf_sim[0]

        else:
            # Get previous feedback and simulated confidence.
            f = feedback[t-1]
            c = int(conf_vec[t-1])

            # Set win boundary
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary

            if f > lower_bound and f < upper_bound:  # win-trial

                # Calculate probability of prediction c (p_pred)
                # and the probabilities of all options  (p_dist)
                conf_sim, p_dist = sim_norm_prob_vectorized(

                                                    np.array([c]),
                                                    np.array([sigma]))

            else:  # lose-trial

                # Calculate probabilities and Negative log likelihood (NLL)
                conf_sim, p_dist = sim_inverted_norm_prob_vectorized(

                                                    np.array([c]),
                                                    np.array([sigma]))

            # Save simulated confidence of trial t
            conf_vec[t] = conf_sim[0]

    return conf_vec


def WSLS_trial(x, *args):

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


def WSLS(x, *args):

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


def bias_model_sim(x, args):

    """
    sim biased model, return simulated confidence.
    mean_option, sigma = x
    n_trials, trial_index = args
    """

    mean_option, sigma = x
    n_trials = args
    conf_sim_vec = np.zeros(n_trials)
    for t in range(n_trials):

        # Calculate probabilities and Negative log likelihood (NLL)
        conf_sim, p_dist = sim_norm_prob_vectorized(np.array([mean_option]),
                                                    np.array([sigma]))
        conf_sim_vec[t] = conf_sim[0]
    return conf_sim_vec


def bias_model_trial(x, *args):
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


def bias_model(x, *args):

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


def process_session(df_s):

    # Participant and session
    participant = df_s.pid.unique()[0]
    session = df_s.session.unique()[0]

    # Remove baseline
    df_s = df_s[df_s.condition != 'baseline']

    # Calculate absolute trial error as the average across subtrials
    df_s['difference'] = abs(df_s['estimate'] - df_s['correct'])
    abs_error_avg = df_s.groupby('trial')['difference'].mean()


    # Only keep first row of every subtrial (one row = one trial)
    df_s = df_s.drop_duplicates(subset='trial', keep='first')

    # Ensure performance aligns with filtered df_s
    # performance is inverse of abs error
    df_s['performance'] = -abs_error_avg.values # df_s['trial'].map(lambda t: -abs_error_avg.loc[t])
  #  print(df_s['performance']), df_s['performance'].iloc[5])
   # print(-abs_error_avg.values == df_s['performance'].values)
    #print("df_s['performance']", df_s['performance'].values)
    # Condition
    conditions = df_s.condition.unique()

    # N trials
    n_trials = len(df_s)

    # Calculate trial-by-trial metrics
    confidence = df_s.confidence.values
    feedback = df_s.feedback.values
    performance = -abs_error_avg.values

    # Initialise dict in which simulation results will be saved
    participant_results = {}

    # Bias model parameters and simulation
    mean_bound = (0, 100)
    sigma_bound = (1, 10)
    bias_params = {
        'mean': np.random.uniform(*mean_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }
    bias_sim_conf = bias_model_sim((bias_params['mean'], bias_params['sigma']), n_trials)
    participant_results.update({'bias_conf': bias_sim_conf, **{f'bias_{k}': v for k, v in bias_params.items()}})

    # WSLS model parameters and simulation
    sigma_bound = (1, 10)
    win_bound = (1, 100)
    wsls_params = {
        'sigma': np.random.uniform(*sigma_bound),
        'win_boundary': np.random.uniform(*win_bound)
    }
    wsls_sim_conf = WSLS_sim((wsls_params['sigma'], wsls_params['win_boundary']), confidence, feedback, n_trials)
    participant_results.update({'wsls_conf': wsls_sim_conf, **{f'wsls_{k}': v for k, v in wsls_params.items()}})

    # RW model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    rw_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }
    confidence_rw = confidence.copy()
    feedback_rw = feedback.copy()
# =============================================================================
#     print('RW Feedback', feedback)
#     print('RW confidence', confidence)
#     print('RW n_trials', n_trials)
# =============================================================================
    rw_sim_conf, y_val_rw = RW_sim((rw_params['alpha'], rw_params['sigma']), confidence, feedback, n_trials)
    participant_results.update({'rw_conf': rw_sim_conf, **{f'rw_{k}': v for k, v in rw_params.items()}})
   # print('rw_sim_conf', rw_sim_conf)

    # RW cond model parameters and simulation
    alpha_neut_bound = (0, 1)
    alpha_pos_bound = (0, 1)
    alpha_neg_bound = (0, 1)
    sigma_bound = (1, 10)
    rw_cond_params = {
        'alpha_neut': np.random.uniform(*alpha_neut_bound),
        'alpha_pos': np.random.uniform(*alpha_pos_bound),
        'alpha_neg': np.random.uniform(*alpha_neg_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }
    rw_cond_sim_conf = RW_sim((rw_cond_params['alpha_neut'], rw_cond_params['alpha_pos'], rw_cond_params['alpha_neg'], rw_cond_params['sigma']), confidence, feedback, n_trials, df_s.condition.values)
    participant_results.update({'rw_cond_conf': rw_cond_sim_conf, **{f'rw_cond_{k}': v for k, v in rw_cond_params.items()}})

    # Choice Kernel model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    beta_bound = (40, 200)
    ck_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'beta': np.random.uniform(*beta_bound)
    }
    choice_kernel_sim_conf = CK_sim((ck_params['alpha'], ck_params['sigma'], ck_params['beta']), confidence, n_trials)
    participant_results.update({'ck_conf': choice_kernel_sim_conf, **{f'ck_{k}': v for k, v in ck_params.items()}})

    # RW + Choice Kernel model parameters and simulation
    alpha_bound = (0, 1)
    alpha_ck_bound = (0, 1)
    sigma_bound = (1, 10)
    sigma_ck_bound = (1, 5)
    beta_bound = (40, 200)
    beta_ck_bound = (40, 200)
    rwck_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'alpha_ck': np.random.uniform(*alpha_ck_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'sigma_ck': np.random.uniform(*sigma_ck_bound),
        'beta': np.random.uniform(*beta_bound),
        'beta_ck': np.random.uniform(*beta_ck_bound)
    }
    rwck_sim_conf = RWCK_sim((rwck_params['alpha'], rwck_params['alpha_ck'], rwck_params['sigma'], rwck_params['sigma_ck'], rwck_params['beta'],  rwck_params['beta_ck']), feedback, confidence, n_trials)
    participant_results.update({'rwck_conf': rwck_sim_conf, **{f'rwck_{k}': v for k, v in rwck_params.items()}})

    # Rescorla-Wagner Performance Delta model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    w_rw_bound = (0, 2)
    w_delta_p_bound = (0, 2) # (2, 8)
    rwpd_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'w_rw': np.random.uniform(*w_rw_bound),
        'w_delta_p': np.random.uniform(*w_delta_p_bound)
    }

    rwpd_sim_conf = RWPD_sim((rwpd_params['alpha'], rwpd_params['sigma'], rwpd_params['w_rw'], rwpd_params['w_delta_p']), confidence, feedback, n_trials, performance)
    participant_results.update({'rwpd_conf': rwpd_sim_conf, **{f'rwpd_{k}': v for k, v in rwpd_params.items()}})

    # Rescorla-Wagner Performance model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    w_rw_bound = (0, 1)
    w_p_bound = (0, 1) # (2, 8)
    rwp_params = {
        'alpha': rw_params['alpha'], #np.random.uniform(*alpha_bound),
        'sigma': rw_params['sigma'], #np.random.uniform(*sigma_bound),
        'w_rw': np.random.uniform(*w_rw_bound),
        'w_p': np.random.uniform(*w_p_bound)
    }

    rwp_sim_conf, y_val_rwp = RWP_sim((rwp_params['alpha'], rwp_params['sigma'], rwp_params['w_rw'], rwp_params['w_p']), confidence, feedback, n_trials, performance)
    participant_results.update({'rwp_conf': rwp_sim_conf, **{f'rwp_{k}': v for k, v in rwp_params.items()}})

    return [participant_results, participant, session]

def main(df):
    # Get unique pid and session pairs
    unique_pairs = df[['pid', 'session']].drop_duplicates()
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

    # Create a list of DataFrames, one for each unique pid-session pair
    df_list = [df[(df['pid'] == pid) & (df['session'] == session)].copy()
               for pid, session in unique_pairs_list]

    # Define model names for easier labeling in the DataFrame
    model_names = [
        "bias", "wsls", "rw", "rw_cond",
        "ck", "rwck", "rwpd", "rwp"
    ]

    # List to collect trial data and model parameters for the final DataFrame
    all_trial_data = []

    # Run the process_session function for each session
    # (i.e., one run for each participant)
    for session_df in df_list:
        # Call process_session and unpack results, participant, and session
        session_results, participant_id, session_id = process_session(session_df)

        # Iterate over each model's results and parameters
        for model_name in model_names:
            # Access confidence data for this model dynamically
            confidence_key = f'{model_name}_conf'
            confidence_data = session_results.get(confidence_key)

            if confidence_data is None:
                print(f"Warning: No confidence data found for model \
                      {model_name} in session results.")
                continue

            # Extract parameters for the current model by filtering out '_conf' keys
            parameters = {k: v for k, v in session_results.items()
                          if k.startswith(model_name)
                          and not k.endswith('_conf')}

            # Create trial data with confidence and parameters
            for trial_idx, confidence in enumerate(confidence_data):
                trial_data = {
                    'confidence_sim': confidence,
                    'model': model_name,
                    'trial': trial_idx + 1,
                    'session': session_id,
                    'pid': participant_id,
                    **parameters
                }
                all_trial_data.append(trial_data)

    # Convert the trial data into a DataFrame
    results_df = pd.DataFrame(all_trial_data)

    return results_df

def plot_model_results(df, df_real):
    """
    Plot results for each model with two columns: individual session outcomes
    and mean confidence over time.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['confidence_sim', 'model', 'trial', 'session', 'pid'].
    """
    # Get unique model names to determine the number of rows
    model_names = df['model'].unique()
    n_models = len(model_names)

    # Set up a 2-column subplot structure
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 4 * n_models),
                             sharex='col', sharey='row')
    if n_models == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    # Plot each model as separate rows
    for model_idx, model_name in enumerate(model_names):
        # Separate the two subplots for this model
        ax_individual = axes[model_idx][0]
        ax_mean = axes[model_idx][1]

        # Filter data for the current model
        model_df = df[df['model'] == model_name]
        model_df_real = df_real[df_real['model'] == model_name]
        # Plot each session's confidence for the current model in the first column
        for session_id in model_df['session'].unique():
            session_df = model_df[model_df['session'] == session_id]
            session_df_real = model_df_real[model_df_real['session'] == session_id]
            ax_individual.plot(session_df['trial'],
                               session_df['confidence_sim'],
                               label=f'Session {session_id}', marker='o')

        # Calculate and plot the mean confidence over trials for the current model
        mean_confidence = model_df.groupby('trial')['confidence_sim'].mean()
        ax_mean.plot(mean_confidence.index, mean_confidence.values, color='b',
                     marker='o', linestyle='-', linewidth=2)

        # Set titles and labels
        ax_individual.set_title(f'{model_name}', fontsize=35)
        ax_mean.set_title('Mean Confidence Over Trials', fontsize=25)
        ax_individual.set_ylabel('Confidence', fontsize=25)
        ax_individual.spines[['top', 'right']].set_visible(False)
        ax_mean.spines[['top', 'right']].set_visible(False)
        ax_individual.set_yticks(range(0,101,20))
        ax_individual.set_yticklabels(range(0,101,20), fontsize=25)
        # Set y-axis limits
        ax_individual.set_ylim(-5, 105)
        ax_mean.set_ylim(-5, 105)
        print(session_df.columns)

    # Label the shared x-axis
    axes[-1][0].set_xlabel('Trial')
    axes[-1][1].set_xlabel('Trial')
    plt.tight_layout()
    plt.show()

def plot_model_results_with_real_data(df, df_real):
    """
    Plot results for each model with three columns: real data, individual session outcomes,
    and mean confidence over time.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['confidence_sim', 'model', 'trial', 'session', 'pid'].
    - df_real (pd.DataFrame): DataFrame with columns ['confidence_real', 'model', 'trial', 'session', 'pid'].
    """
    # Get unique model names to determine the number of rows
    model_names = df['model'].unique()
    n_models = len(model_names)

    # Set up a 3-column subplot structure (real data, individual simulations, mean confidence)
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 4 * n_models),
                             sharex='col', sharey='row')
    if n_models == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    # Plot each model as separate rows
    for model_idx, model_name in enumerate(model_names):
        # Separate the three subplots for this model
        ax_real = axes[model_idx][0]  # Real data plot
        ax_individual = axes[model_idx][1]  # Individual simulations
        ax_mean = axes[model_idx][2]  # Mean confidence

        # Filter data for the current model
        model_df = df[df['model'] == model_name]
        model_df_real = df_real.copy()

        # Plot real data in the first column
        for session_id in model_df_real['session'].unique():
            session_df_real = model_df_real[model_df_real['session'] == session_id]
            ax_real.plot(session_df_real['trial'],
                         session_df_real['confidence'],
                         label=f'Session {session_id}', marker='o',
                         color='C0', alpha=0.7)

        # Plot each session's confidence for the current model in the second column (simulations)
        for session_id in model_df['session'].unique():
            session_df = model_df[model_df['session'] == session_id]
            ax_individual.plot(session_df['trial'],
                               session_df['confidence_sim'],
                               label=f'Session {session_id}', marker='o', alpha=0.7)

        # Calculate and plot the mean confidence over trials for the current model in the third column
        mean_confidence = model_df.groupby('trial')['confidence_sim'].mean()
        mean_confidence_real = model_df_real.groupby('trial')['confidence'].mean()
        ax_mean.plot(mean_confidence_real.index, mean_confidence_real.values, color='black',
                     marker='o', linestyle='-', linewidth=2, label="Real")
        ax_mean.plot(mean_confidence.index, mean_confidence.values, color='b',
                     marker='o', linestyle='-', linewidth=2, label="Simulated")

        # Set titles and labels
        ax_real.set_title(f'Real Data', fontsize=25)
        ax_individual.set_title(f'{model_name}', fontsize=25)
        ax_mean.set_title(f'Mean Confidence Over Trials', fontsize=25)

        ax_real.set_ylabel('Confidence', fontsize=25)
        ax_real.spines[['top', 'right']].set_visible(False)
        ax_individual.spines[['top', 'right']].set_visible(False)
        ax_mean.spines[['top', 'right']].set_visible(False)

        # Set y-axis limits
        ax_real.set_ylim(-5, 105)
        ax_individual.set_ylim(-5, 105)
        ax_mean.set_ylim(-5, 105)

        # Set y-ticks
        for ax in [ax_real, ax_individual, ax_mean]:
            ax.set_yticks(range(0, 101, 20))
            ax.set_yticklabels(range(0, 101, 20), fontsize=25)

    # Label the shared x-axis
    axes[-1][0].set_xlabel('Trial', fontsize=25)
    axes[-1][1].set_xlabel('Trial', fontsize=25)
    axes[-1][2].set_xlabel('Trial', fontsize=25)

    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_estimates_by_pid(df):
    """
    Plots a line plot for each participant (pid) with 'estimate' on the Y-axis
    and 'correct' on the X-axis.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing 'pid', 'estimate', and 'correct' columns.
    """

    # Get unique participant IDs
    unique_pids = df['pid'].unique()

    # create error column
    df['abs_error'] = df.estimate-df.correct

    # Set up the plot
    plt.figure(figsize=(5,5))

    # Plot each participant's data
    for pid in unique_pids:
        pid_df = df[df['pid'] == pid]
        plt.scatter(pid_df['correct'], pid_df['estimate'], marker='o', linestyle='-', label=f'PID {pid}', alpha=0.7)

    # Formatting
    plt.xlabel("Correct", fontsize=12)
    plt.ylabel("Estimate", fontsize=12)
    plt.title("Estimates vs. Correct by Participant", fontsize=14)
    #plt.legend(title="Participant ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.axis('equal')

    # Truth
    plt.plot(pid_df['correct'], pid_df['correct'], marker='o', linestyle='-',  alpha=1, color='k')

    plt.show()


def plot_performance_on_confidence_by_pid(df):
    """

    """

    # Get unique participant IDs
    unique_pids = df['pid'].unique()

    # create error column
    df['abs_error'] = df.estimate-df.correct

    # Set up the plot
    plt.figure(figsize=(5,5))

    # Plot each participant's data
    for pid in unique_pids:
        pid_df = df[df['pid'] == pid]
        plt.scatter(pid_df['abs_error'], pid_df['confidence'], marker='o', linestyle='-', label=f'PID {pid}', alpha=0.7)

    # Formatting
    plt.xlabel("abs_error", fontsize=12)
    plt.ylabel("confidence", fontsize=12)
    plt.title("abs_error vs. confidence by Participant", fontsize=14)
    #plt.legend(title="Participant ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
   # plt.axis('equal')

    # Truth
    #plt.plot(pid_df['correct'], pid_df['correct'], marker='o', linestyle='-',  alpha=1, color='k')

    plt.show()

#-----------------------------------------------------------------------------
# Execution
#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Import data - Variable feedback condition (Experiment 2)
    data_file = r'variable_fb_data_full_processed.csv'
    full_path = data_file
    df = pd.read_csv(full_path, low_memory=False)

    # Add session column
    df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

    # plot_estimates_by_pid(df)
   # plot_performance_on_confidence_by_pid(df)
   #

    # Run main
    results_df = main(df)

    # plot
    #plot_model_results(results_df, df)
    plot_model_results_with_real_data(results_df, df)

    # Merge results_df into df based on 'trial', 'participant', and 'session'
    merged_df = df.merge(results_df, on=['trial', 'pid', 'session'],
                         how='left')

    # Save simulated data
    merged_df.to_csv('simulated_conf_EXP2_data_sigma10.csv',
                     index=False)



