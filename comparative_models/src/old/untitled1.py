# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 00:31:28 2024

@author: carll
"""

# models with static updates (V_pred[t] = V_reported[t-1] + update)


# Models
import numpy as np
from scipy.optimize import minimize
import math
from scipy.ndimage import gaussian_filter1d
from src.utility_functions import (create_probability_distribution,
                                   invert_normal_distribution,
                                   create_truncated_normal)

def big_rw_sim (x, *args):

    """
    Simulates choice and confidence reports.
    """

    alpha, mean_p_belief, std_p_belief, w, s = x
    (n_trials, performance, n_dots, subtrials,
     condition, min_value, max_value) = args

    confidence_pred = np.zeros(n_trials)
    fb_hist_belief = np.zeros(n_trials)
    choice_pred = np.zeros(n_trials*3)

    for t in range(n_trials):

        # loop over the current trial's subtrials
        trial_error = np.zeros(3)
        for st, subtrial in enumerate(subtrials[t]):

            # The number of dots believed to be seen
            perceptual_beleif = n_dots + mean_p_belief

            # Integrate belief with feedback history belief
            if t==0:
                std_integrated = std_p_belief
            else:
                std_integrated = std_p_belief + w*(50-fb_hist_belief[t-1])

            # Make choice based on integrated belief
            ct = t + st # trial + subtrial count
            choice_pred[ct] = round(np.random.normal(perceptual_beleif,
                                                    std_integrated))
            # Compute error
            trial_error[st] = abs(n_dots[ct]-choice_pred[ct])

        # Report confidence: a scalar of std_integrated
        confidence_pred[t] = max(0, min(100, s*std_integrated))

        # Compute feedback
        e = np.mean(trial_error)
        feedback = 100-(((e - min_value) / (max_value - min_value)) * 100)

        # Update feedback history belief
        fb_hist_belief[t] = confidence_pred[t] + alpha*(feedback[t] -
                                                        confidence_pred[t])

    return [choice_pred, confidence_pred, fb_hist_belief]

def big_rw_trial (x, *args):

    """
    Fit confidence reports in dot counting task.
    """

    alpha_plus, alpha_minus, sigma, mean_p_belief, std_p_belief, w, util_func_bias, s = x
    (n_trials,
     n_dots,
     min_value,
     max_value,
     confidence,
     feedback) = args

    confidence_pred = np.zeros(n_trials)
    fb_hist_belief = np.zeros(n_trials)
    choice_pred = np.zeros(n_trials*3)
    std_integrated = np.zeros(n_trials*3)
    sigma_vec = np.full(n_trials, sigma)
    subtrials = range(3)
    ct = -1
    for t in range(n_trials):

        # loop over the current trial's subtrials
        trial_error = np.zeros(3)
        for st in subtrials:
            # Choice trial number
            ct += 1

            # The number of dots believed to be seen
            perceptual_beleif = n_dots[ct] + mean_p_belief

            # Integrate belief with feedback history belief
            if t==0:
                std_integrated[ct] = n_dots[ct]/100
            else:
                #std_integrated = ((1-w)*std_p_belief) + (w*(fb_hist_belief[t-1]))
                #std_p_belief = np.random.normal(std_p_belief, 2)
               # std_integrated[ct] = std_p_belief + w*(50-fb_hist_belief[t-1])
                std_integrated[ct] = ((1-w)*n_dots[ct]) + w*(s*(100-fb_hist_belief[t-1]))
                std_integrated[ct] =  max(0, min(std_integrated[ct], 100))

            # Make choice based on integrated belief
            #choice_pred[ct] = int(np.random.uniform(perceptual_beleif-std_integrated[ct], perceptual_beleif+std_integrated[ct]))
            choice_pred[ct] = int(np.random.normal(perceptual_beleif,
                                                   std_integrated[ct]))

            # Compute error
            trial_error[st] = abs(n_dots[ct]-choice_pred[ct])

        # Report confidence: a scalar of std_integrated
       # conf = 100 - normalize_std_integrated(std_integrated[ct], std_p_belief,
       #                                       w)
       #confidence_pred[t] = max(0, min(conf, 100))
       # confidence_pred[t] = rescale_std_integrated(std_integrated[ct], s)
        if t == 0:
            confidence_pred[t] = 50
        else:
            confidence_pred[t] = max(0, min(fb_hist_belief[t-1], 100)) - s*(n_dots[ct])

        # Compute feedback
      #  if util_func_bias:
      #      feedback[t] = feedback[t] + float(((feedback[t]/100)
      #                                         *util_func_bias))

        if feedback[t] > confidence_pred[t]:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_plus*(feedback[t] -
                                                            confidence_pred[t])
        else:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_minus*(feedback[t] -
                                                            confidence_pred[t])

# =============================================================================
#         # Update sigma_vec based on change in predictors
#         if t == 0:
#             pass
#         else:
#             influence_delta = (fb_hist_belief[t-1]-fb_hist_belief[t]) + (n_dots[ct-1]-n_dots[ct])
#             sigma_vec[t] = max(0, max(sigma + influence_delta, 50))
# =============================================================================

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial trial
    confidence_pred_ = confidence_pred[1:]
    sigma_vec_ = sigma_vec[1:]
    confidence_ = confidence[1:]

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(confidence_pred_,  # ignore first trial
                                           sigma_vec_,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence_)
                   + epsilon)

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1 (1 item already removed)
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 when indexing from 1 (2 items already removed)
        nlls = np.delete(nlls, 38)

    return [nlls, sigma_vec, choice_pred, confidence_pred, fb_hist_belief]

def normalize_std_integrated(std_integrated, std_p_belief, w):
    min_std_integrated = std_p_belief - 50 * w
    max_std_integrated = std_p_belief + 50 * w
    normalized_std_integrated = ((std_integrated - min_std_integrated) / (max_std_integrated - min_std_integrated)) * 100
    return normalized_std_integrated

def normalize_to_0_100(data, max_, min_):
    min_data = min_
    max_data = max_
    normalized_data = ((data - min_data) / (max_data - min_data)) * 100
    return normalized_data

def rescale_std_integrated(std_integrated, s):
    normalized_std_integrated = std_integrated * s
    return normalized_std_integrated

def big_rw (x, *args):

    """
    Fit confidence reports in dot counting task.
    """

    (alpha_plus, alpha_minus, sigma,
     mean_p_belief, std_p_belief, w, util_func_bias, s) = x
    (n_trials,
     n_dots,
     min_value,
     max_value,
     confidence,
     feedback) = args

    confidence_pred = np.zeros(n_trials)
    fb_hist_belief = np.zeros(n_trials)
    choice_pred = np.zeros(n_trials*3)
    std_integrated = np.zeros(n_trials*3)
    sigma_vec = np.full(n_trials, sigma)
    subtrials = range(3)
    ct = -1
    for t in range(n_trials):

        # loop over the current trial's subtrials
        trial_error = np.zeros(3)
        for st in subtrials:
            # Choice trial number
            ct += 1

            # The number of dots believed to be seen
            perceptual_beleif = n_dots[ct] + mean_p_belief

            # Integrate belief with feedback history belief
            if t==0:
                std_integrated[ct] = n_dots[ct]/100
            else:
                #std_integrated = ((1-w)*std_p_belief) + (w*(fb_hist_belief[t-1]))
                #std_p_belief = np.random.normal(std_p_belief, 2)
                std_integrated[ct]  = ((1-w)*n_dots[ct]) + w*(s*(100-fb_hist_belief[t-1]))
                #std_integrated[ct] = std_p_belief + w*(50-fb_hist_belief[t-1])
                std_integrated[ct] =  max(0, min(std_integrated[ct], 100))

            # Make choice based on integrated belief
            #choice_pred[ct] = int(np.random.uniform(perceptual_beleif-std_integrated[ct], perceptual_beleif+std_integrated[ct]))
            choice_pred[ct] = int(np.random.normal(perceptual_beleif, std_integrated[ct]))

            # Compute error
            trial_error[st] = abs(n_dots[ct]-choice_pred[ct])

        # Report confidence: a scalar of std_integrated
        #conf = 100 - normalize_std_integrated(std_integrated[ct], std_p_belief, w)
        if t == 0:
            confidence_pred[t] = 50
        else:
            confidence_pred[t] = max(0, min(fb_hist_belief[t-1], 100)) - s*(n_dots[ct])
        #confidence_pred[t] =  #100-std_integrated[ct] #rescale_std_integrated(std_integrated[ct], s)

        # Compute feedback
       # if util_func_bias:
       #     feedback[t] = feedback[t] + float(((feedback[t]/100)
       #                                        *util_func_bias))

        if feedback[t] > confidence_pred[t]:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_plus*(feedback[t] -
                                                            confidence_pred[t])
        else:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_minus*(feedback[t] -
                                                            confidence_pred[t])
# =============================================================================
#         # Update sigma_vec based on change in predictors
#         if t == 0:
#             pass
#         else:
#             influence_delta = (fb_hist_belief[t-1]-fb_hist_belief[t]) + (n_dots[ct-1]-n_dots[ct])
#             sigma_vec[t] = max(0, max(sigma + influence_delta, 50))
#
# =============================================================================
    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial trial
    confidence_pred = confidence_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(confidence_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1 (1 item already removed)
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 when indexing from 1 (2 items already removed)
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)

def delta_P_RW_sim (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the change in performance
    since the last trial (delta P).

    return list of vectors
    """

    alpha, sigma, bias, w_RW, w_Delta_P = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias

        else:

            # If concatenating sessions...
            # ...reset c at new session
            if t == 20 or t == 40:
                c = bias
            else:
                #c = int(c_rw[t-1])
                c = int(confidence[t-1])

            # Get previous confidence value and feedback,
            f = feedback[t-1]
           # c = int(confidence[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Encure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get delta performance
            delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between 0 and 100.
            delta_p = max(0, min(100, delta_p))

            # Get the prediction as the weighted sum
            model_pred[t] = (w_RW*c_rw[t]) + (w_Delta_P*delta_p)

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]

def delta_P_RW_trial (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the change in performance
    since the last trial (delta P).

    return list of vectors
    """

    alpha, sigma, bias, w_RW, w_Delta_P = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias

        else:

            # If concatenating sessions...
            # ...reset c at new session
            if t == 20 or t == 40:
                c = bias
            else:
                #c = int(c_rw[t-1])
                c = int(confidence[t-1])


            # Get previous confidence value and feedback,
            f = feedback[t-1]
           # c = int(confidence[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Encure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get delta performance
            delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between 0 and 100.
            delta_p = max(0, min(100, delta_p))

            # Get the prediction as the weighted sum
            model_pred[t] = (w_RW*c_rw[t]) + (w_Delta_P*delta_p)

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]

def delta_P_RW (x, *args):

    """
    Confidence is updated as the weighted sum of: (1)
    confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
    the change in performance since the last trial (delta P).
    """

    alpha, sigma, bias, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias
        else:

            # If concatenating sessions...
            # ...reset c at new session
            if t == 20 or t == 40:
                c = bias
            else:
                #c = int(c_rw[t-1])
                c = int(confidence[t-1])

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            # c = int(confidence[t-1])


            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Encure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance delta
            delta_p = performance[t] - performance[t-1]

            # Encure delta_p is between -100 and 100.
            delta_p = max(-100, min(100, delta_p))

            # Get the prediction as the weighted sum
            model_pred[t] = (w_RW*c_rw[t]) + (w_PD*delta_p)

    # Small value to avoid division by 0
    epsilon = 1e-10

    # Remove initial baseline trial
    model_pred = model_pred[1:]
    sigma_vec = sigma_vec[1:]
    confidence = confidence[1:]

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1 (1 item already removed)
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 when indexing from 1 (2 items already removed)
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)

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

    alpha, alpha_ck, sigma, bias, beta, beta_ck = x
    feedback, confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)
    model_pred = np.zeros(n_trials)
    model_probs = []

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

            # If concatenating sessions...
            # ...reset choice kernels and c_pred at new session
            if t == 20 or t == 40:
                choice_kernels = np.full(array_length, value)
                c = bias
            else:
                #c = model_pred[t-1]
                c = int(confidence[t-1])

            # Update "value" --------------------------------------------

            # Get previous confidence value and feedback,
            f = feedback[t-1]

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))

            # Get option probs from truncated normal distribution
            options = np.arange(0,101, 1)
            p_options = create_truncated_normal(np.array([c_pred]),
                                                np.array([sigma]),
                                                lower_bound=0,
                                                upper_bound=100).pdf(options)

            # -----------------------------------------------------------

            # Update choice kernels
            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha_ck * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma)

            # -----------------------------------------------------------

            # Combine RW update and CK update
            p = np.zeros(len(p_options))
            for i, (v_k, ck_k) in enumerate(zip(p_options,
                                                smoothed_choice_kernels)):

                # Exponential of each beta * value + beta + choice kernel
                p[i] = np.exp((beta*v_k) + (beta_ck*ck_k))

            # Normalize to get probabilities
            probabilities = p / np.sum(p)
            model_probs.append(probabilities)

            # Get nll
            nlls[t] = -np.log(probabilities[int(confidence[t])])

    # Remove initial baseline trial
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

    alpha, alpha_ck, sigma, bias, beta, beta_ck = x
    feedback, confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)
    model_pred = np.zeros(n_trials)
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

            # If concatenating sessions...
            # ...reset choice kernels and c_pred at new session
            if t == 20 or t == 40:
                choice_kernels = np.full(array_length, value)
                c = bias
            else:
                #c = model_pred[t-1]
                c = int(confidence[t-1])

            # Update "value" --------------------------------------------

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            #c = int(confidence[t-1])

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

            # Get option probs from truncated normal distribution
            options = np.arange(0,101, 1)
            p_options = create_truncated_normal(np.array([c_pred]),
                                                np.array([sigma]),
                                                lower_bound=0,
                                                upper_bound=100).pdf(options)

            # -----------------------------------------------------------
            # Update choice kernels
            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha_ck * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma)

            # -----------------------------------------------------------

            # Combine RW update and CK update
            p = np.zeros(len(p_options))
            for i, (v_k, ck_k) in enumerate(zip(p_options,
                                                smoothed_choice_kernels)):

                # Exponential of each beta * value + beta_ck + choice kernel
                p[i] = np.exp((beta*v_k) + (beta_ck*ck_k))

            # Normalize to get probabilities
            probabilities = p / np.sum(p)

            # Get nll
            nlls[t] = -np.log(probabilities[int(confidence[t])])

    # Remove initial baseline trial
    nlls = nlls[1:]

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)


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
            nlls[t] = -np.log(probabilities[int(confidence[t])])

        if t > 0:

            # If concatenating sessions, reset choice kernels at new session
            if t == 20 or t == 40:
                choice_kernels = np.full(array_length, value)

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
            nlls[t] = -np.log(probabilities[int(confidence[t])])

    # Remove initial baseline trial
    nlls = nlls[1:]
    model_pred = model_pred[1:]
    model_probs = model_probs[1:]

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
    confidence, n_trials = args

    array_length = 101
    value = 1 / array_length
    choice_kernels = np.full(array_length, value)
    nlls = np.zeros(n_trials)
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

            # If concatenating sessions, reset choice kernels at new session
            if t == 20 or t == 40:
                choice_kernels = np.full(array_length, value)

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
            nlls[t] = -np.log(probabilities[int(confidence[t])])

    # Remove initial baseline trial
    nlls = nlls[1:]

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)



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

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])  # -1 to ensure 0 indexing
            #c = model_pred[t-1]

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

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

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
    confidence, feedback, n_trials, condition = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = bias

# =============================================================================
#             # Due to lack of previous feedback, set PE to 0
#             PE = 0  # Prediction error
#             c = int(confidence[t-1]) # Previous confidence
#             c_pred = c + alpha_neut*PE  # Update rule
#             model_pred[t] = c_pred
# =============================================================================

        if t > 0:

            # The feedback condition on the previous trial dictates the alpha
            if condition[t-1] == 'neut':
                alpha = alpha_neut
            if condition[t-1] == 'pos':
                alpha = alpha_pos
            if condition[t-1] == 'neg':
                alpha = alpha_neg

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])
            #c = model_pred[t-1]

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

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)


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

# =============================================================================
#             # Due to lack of previous feedback, set PE to 0
#             PE = 0  # Prediction error
#             c = int(confidence[t-1]) # Previous confidence
#             c_pred = c + alpha*PE  # Update rule
#             model_pred[t] = c_pred
# =============================================================================

        if t > 0:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])
           # c = model_pred[t-1]

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

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

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
    confidence, feedback, n_trials = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = bias

# =============================================================================
#             # Due to lack of previous feedback, set PE to 0
#             PE = 0  # Prediction error
#             c = int(confidence[t-1]) # Previous confidence
#             c_pred = c + alpha*PE  # Update rule
#             model_pred[t] = c_pred
# =============================================================================


        if t > 0:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])
            #c = model_pred[t-1]

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

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)


def rw_static_LR_trial(x, *args):
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

# =============================================================================
#             # Due to lack of previous feedback, set PE to 0
#             PE = 0  # Prediction error
#             c = int(confidence[t-1]) # Previous confidence
#             c_pred = c + alpha*PE  # Update rule
#             model_pred[t] = c_pred
# =============================================================================

        if t > 0:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])  # -1 to ensure 0 indexing

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

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    return [nlls, model_pred, sigma_vec, confidence]


def rw_static(x, *args):

    """
    Does not track confidence, only updates with reported confidence.

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
    confidence, feedback, n_trials = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_pred[t] = bias

# =============================================================================
#             # Due to lack of previous feedback, set PE to 0
#             PE = 0  # Prediction error
#             c = int(confidence[t-1]) # Previous confidence
#             c_pred = c + alpha*PE  # Update rule
#             model_pred[t] = c_pred
# =============================================================================


        if t > 0:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])
           # c = model_pred[t-1] # For continous

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

    # Get NLLs from truncated probability distribution
    nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                           sigma_vec,
                                           lower_bound=0,
                                           upper_bound=100).pdf(confidence)
                   + epsilon)

    # Remove first trial in concatenated sessions
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nlls = np.delete(nlls, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nlls = np.delete(nlls, 38)

    return np.sum(nlls)

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

# =============================================================================
#         if t == 1:
#             # Keep to previous confidence
#             c = int(confidence[t-1])
#
#             # Get option probabilities
#             prob_distribution = create_probability_distribution(options,
#                                                                 c,
#                                                                 sigma)
#
#             prob_dists.append(prob_distribution)
#
#             # Get negative log likelihood of reported confidence
#             log_likelihood = np.log(prob_distribution[confidence[t]]
#                                     + epsilon)
#             nll_vector[t] = -log_likelihood
# =============================================================================

        else:

            # Get previous confidence value and set
            f = feedback[t-1]
            c = int(confidence[t-1])

            # Set win boundary
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary

            if f > lower_bound and f < upper_bound:  # win-trial

                # Get option probabilities
                prob_distribution = create_probability_distribution(options,
                                                                    c,
                                                                    sigma)

                prob_dists.append(prob_distribution)
            else:  # lose-trial

                # Get probability function
                prob_dist = create_probability_distribution(options,
                                                            c,
                                                            sigma)

                # inverted distribution
                prob_distribution = invert_normal_distribution(prob_dist)
                prob_dists.append(prob_distribution)

            # Get negative log likelihood of reported confidence
            log_likelihood = np.log(prob_distribution[confidence[t]]
                                    + epsilon)
            nll_vector[t] = -log_likelihood

    # remove baseline
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
    confidence, feedback, n_trials = args

    options = np.linspace(0, 100, 100+1)
    nll_vector = np.zeros(n_trials)

    for t in range(n_trials):

        epsilon = 1e-10
        if t == 0:  # first trial, only included to index previous c and f.
            continue

# =============================================================================
#         if t == 1:
#             # Keep to previous confidence
#             c = int(confidence[t-1])
#
#             # Get option probabilities
#             prob_distribution = create_probability_distribution(options,
#                                                                 c,
#                                                                 sigma)
#             # Get negative log likelihood of confidence
#             log_likelihood = np.log(prob_distribution[int(confidence[t])]
#                                     + epsilon)
#             nll_vector[t] = -log_likelihood
# =============================================================================

        else:
            # Get previous confidence value and set
            f = feedback[t-1]
            c = int(confidence[t-1])

            # Set win boundary
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary

            if f > lower_bound and f < upper_bound:  # Win-trial

                # Get option probabilities
                prob_distribution = create_probability_distribution(options,
                                                                    c,
                                                                    sigma)

            else:  # Lose-trial

                # Get probability function
                prob_dist = create_probability_distribution(options,
                                                            c,
                                                            sigma)

                # Inverted distribution
                prob_distribution = invert_normal_distribution(prob_dist)

            # Get negative log likelihood of confidence
            log_likelihood = np.log(prob_distribution[int(confidence[t])]
                                    + epsilon)
            nll_vector[t] = -log_likelihood

    # Remove first trial in session
    if n_trials <= 20:
        # remove first trial to
        nll_vector = nll_vector[1:]
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nll_vector = np.delete(nll_vector, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nll_vector = np.delete(nll_vector, 38)

    return np.sum(nll_vector)


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

    options = np.linspace(0, 100, 100+1)
    nll_vector = np.zeros(n_trials)
    prob_dists = []
    for t in range(n_trials):
        if t == 0:  # First trial, nll not included for this trial
            continue
        else:
            # Probability distribution
            prob_distribution = create_probability_distribution(options,
                                                                mean_option,
                                                                sigma=sigma)
            prob_dists.append(prob_distribution)

            # Get current confidence score
            c = int(confidence[t])

            # Get negative log likelihood of choosing that confidence
            epsilon = 1e-10
            log_likelihood = np.log(prob_distribution[c] + epsilon)
            nll_vector[t] = -log_likelihood

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
    confidence, n_trials = args

    options = np.linspace(0, 100, 100+1)
    nll_vector = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:  # first trial, only included to index passed c and f.
            continue
        else:

            # Probability distribution
            # Get option probabilities
            prob_distribution = create_probability_distribution(options,
                                                                mean_option,
                                                                sigma=sigma)

            # Get current confidence score
            c = int(confidence[t])

            # Get negative log likelihood of choosing that confidence
            epsilon = 1e-10
            log_likelihood = np.log(prob_distribution[c] + epsilon)
            nll_vector[t] = -log_likelihood

    if n_trials <= 20:
        # remove first trial to
        nll_vector = nll_vector[1:]
    if n_trials > 20:
        # remove first trial in second session
        # 19 is trial 21 when indexing from 1
        nll_vector = np.delete(nll_vector, 19)
    if n_trials > 40:
        # remove first trial in second session
        # 38 is trial 41 due when indexing from 1
        nll_vector = np.delete(nll_vector, 38)
    #print(f'biased model nll vector length: {len(nll_vector)}')
    return np.sum(nll_vector)


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
    if n_trials <= 20:
        n_trials = n_trials-1
    if n_trials > 20:
        n_trials = n_trials-1
    if n_trials > 40:
        n_trials = n_trials-1

    random_choice_probability = 1/prediction_range
    log_likelihood = math.log(random_choice_probability)
    nll_vector = [-log_likelihood]*n_trials

    return np.sum(nll_vector)


def normalize(params, bounds):
    return [(p - b[0]) / (b[1] - b[0]) for p, b in zip(params, bounds)]

def inverse_normalize(norm_params, bounds):
    return [p * (b[1] - b[0]) + b[0] for p, b in zip(norm_params, bounds)]

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
        nll_model = nll
        n = n_trials
        nll_null = -n * np.log(0.5)
        pseudo_r2 = 1 - (nll_model / nll_null)

        # AIC and BIC
        k = len(best_params)
        aic = 2*k + 2*nll
        bic = k*np.log(n) + 2*nll

        # Append to result and nll lists
        result_list.append(best_params + [nll, aic, bic, pseudo_r2])
        nll_list.append(nll)

    # Index best result
    best_result_index = nll_list.index(min(nll_list))
    best_result = result_list[best_result_index]

    return best_result


# =============================================================================
# def fit_model(model, args, bounds, n_trials, start_value_number=10,
#               solver="L-BFGS-B"):
#     """
#     Fits a model using the minimize function with various start values.
#
#     This function optimizes a model's parameters by exploring multiple
#     starting points within specified bounds. It uses scipy.optimize's
#     minimize function and selects the best result based on the lowest
#     NLL (negative log likelihood). It also calculates additional
#     statistical measures like Akaike Information Criterion (AIC),
#     Bayesian Information Criterion (BIC), and McFadden's pseudo
#     R-squared.
#
#     Parameters:
#     model (callable): The model function to optimize.
#     args (tuple): Arguments to pass to the model function.
#     bounds (list of tuples): Parameter bounds in the format (min, max).
#     n_trials (int): Number of trials in the dataset.
#     start_value_number (int, optional): Number of start values for each
#                                         parameter. Default is 10.
#     solver (str, optional): Optimization algorithm. Default is "L-BFGS-B".
#
#     Returns:
#     list: Best-fitting parameters, NLL, AIC, BIC, and McFadden's
#           pseudo R-squared.
#     """
#
#     # Get start ranges for parameters
#     start_ranges = []
#     for bound in bounds:
#         lower_bound, upper_bound = bound
#         start_range = np.random.uniform(lower_bound,
#                                         upper_bound,
#                                         size=start_value_number)
#         start_ranges.append(start_range)
#
#     # Loop over start values
#     result_list = []
#     nll_list = []
#     for start_values in zip(*start_ranges):
#
#         result = minimize(model,
#                           start_values,
#                           args=args,
#                           bounds=bounds,
#                           method=solver)
#
#         # Get results
#         best_params = result.x
#         nll = result.fun
#
#         # McFadden pseudo r2
#         nll_model = nll
#         n = n_trials
#         nll_null = -n * np.log(0.5)
#         pseudo_r2 = 1 - (nll_model / nll_null)
#
#         # Get AIC and BIC
#         k = len(start_values)  # Number of parameters fitted
#         aic = 2*k + 2*nll
#         bic = k*np.log(n) + 2*nll
#
#         # Append to result and nll lists
#         result_list.append(list(best_params) + [nll, aic, bic, pseudo_r2])
#         nll_list.append(nll)
#
#     # Return best result
#     best_result_index = nll_list.index(min(nll_list))
#     best_result = result_list[best_result_index]
#
#     return best_result
# =============================================================================
