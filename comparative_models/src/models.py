# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 00:01:22 2025

@author: carll
"""

# Models
import numpy as np
from scipy.ndimage import gaussian_filter1d
import math
from src.utils import (calc_norm_prob_vectorized,
                       calc_inverted_norm_prob_vectorized,
                       sim_norm_prob_vectorized,
                       sim_inverted_norm_prob_vectorized)

def LMFP_sim (x, *args):

    """
    Linear weighted sum of intecept and current performance and previous
    feedback. Performance is the negative absolute error.
    """

    sigma, intercept, wp, wf = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_lmfp = np.zeros(n_trials)
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

            # Update rule
            c_lmfp[t] = intercept + wp*performance[t] + wf*feedback[t-1]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmfp[t]))


            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]

    return conf_vec

def LMFP_trial (x, *args):

    """
    Linear weighted sum of intecept and current performance and previous
    feedback. Performance is the negative absolute error.
    """

    sigma, bias, intercept, wp, wf = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_lmfp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_lmfp[t] = bias

        else:

            # Update rule
            c_lmfp[t] = intercept + wp*performance[t] + wf*feedback[t-1]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmfp[t]))



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

def LMFP (x, *args):

    """
    Linear weighted sum of intecept and current performance and previous
    feedback. Performance is the negative absolute error.
    """

    sigma, bias, intercept, wp, wf = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_lmfp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_lmfp[t] = bias
        else:

            # Update rule
            c_lmfp[t] = intercept + wp*performance[t] + wf*feedback[t-1]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmfp[t]))

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

def LMP_sim (x, *args):

    """
    Linear weighted sum of intecept and current performance. Performance is
    the negative absolute error.
    """

    sigma, intercept, wp = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_lmp = np.zeros(n_trials)
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

            # Update rule
            c_lmp[t] = intercept + wp*performance[t]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmp[t]))


            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]

    return conf_vec

def LMP_trial (x, *args):

    """
    Linear weighted sum of intecept and current performance. Performance is
    the negative absolute error.
    """

    sigma, bias, intercept, wp = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_lmp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_lmp[t] = bias

        else:

            # Update rule
            c_lmp[t] = intercept + wp*performance[t]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmp[t]))



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

def LMP (x, *args):

    """
    Linear weighted sum of intecept and current performance. Performance is
    the negative absolute error.
    """

    sigma, bias, intercept, wp = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_lmp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_lmp[t] = bias
        else:

            # Update rule
            c_lmp[t] = intercept + wp*performance[t]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmp[t]))

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

def LMF_sim (x, *args):

    """
    Linear weighted sum of intecept and previous feedback.
    """

    sigma, intercept, wf = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_lmf = np.zeros(n_trials)
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

            # Update rule
            c_lmf[t] = intercept + wf*feedback[t-1]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmf[t]))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]

    return conf_vec

def LMF_trial (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the change in performance
    since the last trial (delta P).

    return list of vectors
    """

    sigma, bias, intercept, wf = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_lmf = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_lmf[t] = bias

        else:

            # Update rule
            c_lmf[t] = intercept + wf*feedback[t-1]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmf[t]))


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

def LMF (x, *args):

    """
    Confidence is updated as the weighted sum of: (1)
    confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
    the averge performance (P) on the last 3 subtrials.
    """

    sigma, bias, intercept, wf = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_lmf = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_lmf[t] = bias
        else:

            # Update rule
            c_lmf[t] = intercept + wf*feedback[t-1]

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_lmf[t]))

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

def RWFP_sim (x, *args):

    """
    RW model where the outcome is a weighted sum of current performance and
    previous feedback. Performance is the negative absolute error.
    """

    alpha, sigma, wf, wp = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_rwfp = np.zeros(n_trials)
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

            # Update rule
            c_rwfp[t] = model_pred[t-1] + alpha*((wp*performance[t] + wf*feedback[t-1]) - model_pred[t-1])

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_rwfp[t]))


            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]

    return conf_vec

def RWFP_trial (x, *args):

    """
    RW model where the outcome is a weighted sum of current performance and
    previous feedback. Performance is the negative absolute error.
    """

    alpha, sigma, bias, wf, wp = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rwfp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rwfp[t] = bias

        else:

            # Update rule
            c_rwfp[t] = model_pred[t-1] + alpha*((wp*performance[t] + wf*feedback[t-1]) - model_pred[t-1])

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_rwfp[t]))


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

def RWFP (x, *args):

    """
    RW model where the outcome is a weighted sum of current performance and
    previous feedback. Performance is the negative absolute error.
    """

    alpha, sigma, bias, wf, wp = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rwfp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rwfp[t] = bias
        else:

            # Update rule
            c_rwfp[t] = model_pred[t-1] + alpha*((wp*performance[t] + wf*feedback[t-1]) - model_pred[t-1])

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_rwfp[t]))

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

def RWP_sim (x, *args):


    """
    RW model where the outcome is the current performance.
    Performance is the negative absolute error.
    """

    alpha, sigma, wp = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_rwp = np.zeros(n_trials)
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
            # Scale performance: 100 = Perfect; 0 = 100 point error
            performance_scaled = max(0, min(100 + performance[t], 100))

            # Update rule
            c_rwp[t] = model_pred[t-1] + alpha*(wp*performance_scaled - model_pred[t-1])

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_rwp[t]))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]

    return conf_vec

def RWP_trial (x, *args):

    """
    RW model where the outcome is the current performance.
    Performance is the negative absolute error.
    """

    alpha, sigma, bias, wp = x
    confidence, feedback, n_trials, performance = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rwp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rwp[t] = bias

        else:

            # Update rule
            c_rwp[t] = model_pred[t-1] + alpha*(wp*performance[t] - model_pred[t-1])

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_rwp[t]))

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
    RW model where the outcome is the current performance.
    Performance is the negative absolute error.
    """

    alpha, sigma, bias, wp = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rwp = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rwp[t] = bias
        else:

            # Update rule
            c_rwp[t] = model_pred[t-1] + alpha*(wp*performance[t] - model_pred[t-1])

            # Ensure the model prediction is between 0 and 100.
            model_pred[t] = max(0, min(100, c_rwp[t]))

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

# =============================================================================
#
# def RWP_sim (x, *args):
#
#     """
#     Sim RWP model returning simulated confidence estimates
#     alpha, sigma, w_RW, w_PD = x
#     confidence, feedback, n_trials, performance = args
#     """
#
#     alpha, sigma, w_RW, w_P, intercept = x
#     confidence, feedback, n_trials, performance = args
#
#     model_pred = np.zeros(n_trials)
#     c_rw = np.zeros(n_trials)
#     conf_vec = np.zeros(n_trials)
#     for t in range(n_trials):
#
#         if t == 0:
#             # The probability mean for the first trial
#             model_pred[t] = confidence[t]
#
#             # Calculate probabilities and simulate confidence estimate
#             conf_sim, y_values = sim_norm_prob_vectorized(
#                                                     np.array([model_pred[t]]),
#                                                     np.array([sigma]))
#
#             # Save simulated estimate to confidence array
#             conf_vec[t] = conf_sim[0]
#
#         else:
#             # Get previous feedback (f) and confidence estimate (c)
#             f = feedback[t-1]
#             c = model_pred[t-1] #c_rw[t-1]
#
#             # Update rule
#             c_rw[t]  = c + alpha*(f - c)
#
#             # Ensure c_rw is between 0 and 100.
#             c_rw[t] = max(0, min(100, c_rw[t]))
#
#             # Get performance influence on confidence
#             # The +100 establish a threshold for when the error size should be
#             # considered so large that it can be ignored.
#             c_P = performance[t] + 100
#
#             model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_P*c_P)))
#
#             # Calculate probabilities across different options (p_options)
#             conf_sim, p_options = sim_norm_prob_vectorized(
#                                                      np.array([model_pred[t]]),
#                                                      np.array([sigma]),
#                                                      )
#
#             conf_vec[t] = conf_sim[0]
#
#     return conf_vec, p_options
#
# def RWP_trial (x, *args):
#
#     """
#     Confidence is updated as the weighted sum of: (1) confidence updated by a
#     rescorla-wagner updated rule (C_RW) and (2) the change in performance
#     since the last trial (delta P).
#
#     return list of vectors
#     """
#
#     alpha, sigma, bias, w_RW, w_P, intercept = x
#     confidence, feedback, n_trials, performance = args
#
#     sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
#     model_pred = np.zeros(n_trials)
#     c_rw = np.zeros(n_trials)
#     # Small value to avoid division by 0
#     epsilon = 1e-10
#     for t in range(n_trials):
#
#         if t == 0:
#             # The probability mean for the first trial (bias) is fit
#             # Trial included to get previous feedback and performance
#             model_pred[t] = bias
#             c_rw[t] = bias
#
#         else:
#
#             # Get previous confidence value and feedback,
#             f = feedback[t-1]
#             c = int(c_rw[t-1])
#
#             PE = f - c  # Prediction error
#             c_rw[t] = c + alpha*PE  # Update rule
#
#             # Encure c_rw is between 0 and 100.
#             c_rw[t] = max(0, min(100, c_rw[t]))
#
#             # Get performance influence on confidence
#             # The +100 establish a threshold for when the error size should be
#             # considered so large that it can be ignored.
#             c_P = performance[t] + 100
#
#             # Get the prediction as the weighted sum
#             model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_P*c_P)))
#
#     # Remove initial baseline trial
#     model_pred = model_pred[1:]
#     sigma_vec = sigma_vec[1:]
#     confidence = confidence[1:]
#
#     # Calculate probabilities and Negative log likelihood (NLL)
#     probabilities, y_values = calc_norm_prob_vectorized(confidence,
#                                                         model_pred,
#                                                         sigma_vec)
#     nlls = -np.log(probabilities + epsilon)
#
#     return [nlls, model_pred, sigma_vec, confidence]
#
# def RWP (x, *args):
#
#     """
#     Confidence is updated as the weighted sum of: (1)
#     confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
#     the averge performance (P) on the last 3 subtrials.
#     """
#
#     alpha, sigma, bias, w_RW, w_P, intercept = x
#     confidence, feedback, n_trials, performance, trial_index = args
#
#     sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
#     model_pred = np.zeros(n_trials)
#     c_rw = np.zeros(n_trials)
#     # Small value to avoid division by 0
#     epsilon = 1e-10
#     for t in range(n_trials):
#
#         if t == 0:
#             # The probability mean for the first trial (bias) is fit
#             # Trial included to get previous feedback and performance
#             model_pred[t] = bias
#             c_rw[t] = bias
#         else:
#             # Get previous feedback (f) and confidence estimate (c)
#             f = feedback[t-1]
#             c = int(c_rw[t-1])
#
#             PE = f - c  # Prediction error
#             c_rw[t] = c + alpha*PE  # Update rule
#
#             # Ensure c_rw is between 0 and 100.
#             c_rw[t] = max(0, min(100, c_rw[t]))
#
#             # Get performance influence on confidence
#             # The +100 establish a threshold for when the error size should be
#             # considered so large that it can be ignored.
#             c_P = performance[t] + 100
#
#             model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_P*c_P)))
#
#     # Remove initial baseline trial
#     model_pred = model_pred[1:]
#     sigma_vec = sigma_vec[1:]
#     confidence = confidence[1:]
#
#     # Calculate probabilities and Negative log likelihood (NLL)
#     probabilities, y_values = calc_norm_prob_vectorized(confidence,
#                                                         model_pred,
#                                                         sigma_vec)
#     nlls = -np.log(probabilities + epsilon)
#
#     return np.sum(nlls[trial_index])
# =============================================================================


def RWPD_simv2 (x, *args):

    """
    Sim RWPD model retuning simulated confidence estimates
    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args
    """

    alpha, sigma, bias, w_RW, w_PD, intercept, alpha_p = x
    confidence, feedback, n_trials, performance, trial_index = args

    model_pred = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    c_pd = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias
            c_pd[t] = bias
        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get confidence estimate based on performance delta
            c_pd[t] = c_pd[t-1] + alpha_p*(performance[t] - c_pd[t-1])

            # Encure c_p[t] is between -100 and 100.
            c_pd[t] = max(-100, min(100, c_pd[t]))

            model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_PD*c_pd[t])))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     lower_bound=0,
                                                     upper_bound=100)
            conf_vec[t] = conf_sim[0]

    return conf_vec

def RWPD_trialv2 (x, *args):

    """
    Confidence is updated as the weighted sum of: (1) confidence updated by a
    rescorla-wagner updated rule (C_RW) and (2) the change in performance
    since the last trial (delta P).

    return list of vectors
    """

    alpha, sigma, bias, w_RW, w_PD, intercept, alpha_p = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    c_pd = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias
            c_pd[t] = bias
        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get confidence estimate based on performance delta
            c_pd[t] = c_pd[t-1] + alpha_p*(performance[t] - c_pd[t-1])

            # Encure c_p[t] is between -100 and 100.
            c_pd[t] = max(-100, min(100, c_pd[t]))

            model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_PD*c_pd[t])))

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

def RWPDv2 (x, *args):

    """
    Confidence is updated as the weighted sum of: (1)
    confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
    the change in performance since the last trial (delta P).
    """

    alpha, sigma, bias, w_RW, w_PD, intercept, alpha_p = x
    confidence, feedback, n_trials, performance, trial_index = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    c_pd = np.zeros(n_trials)
    # Small value to avoid division by 0
    epsilon = 1e-10
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial (bias) is fit
            # Trial included to get previous feedback and performance
            model_pred[t] = bias
            c_rw[t] = bias
            c_pd[t] = bias
        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get confidence estimate based on performance delta
            c_pd[t] = c_pd[t-1] + alpha_p*(performance[t] - c_pd[t-1])

            # Encure c_p[t] is between -100 and 100.
            c_pd[t] = max(-100, min(100, c_pd[t]))

            model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_PD*c_pd[t])))

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

# =============================================================================
#
# def RWPD_sim (x, *args):
#
#     """
#     Sim RWPD model retuning simulated confidence estimates
#     alpha, sigma, w_RW, w_PD = x
#     confidence, feedback, n_trials, performance = args
#     """
#
#     alpha, sigma, w_RW, w_PD, intercept = x
#     confidence, feedback, n_trials, performance = args
#
#     model_pred = np.zeros(n_trials)
#     c_rw = np.zeros(n_trials)
#     conf_vec = np.zeros(n_trials)
#     for t in range(n_trials):
#
#         if t == 0:
#             # The probability mean for the first trial
#             model_pred[t] = confidence[t]
#             c_rw[t] = confidence[t]
#             conf_vec[t] = confidence[t]
#         else:
#             # Get previous feedback (f) and confidence estimate (c)
#             f = feedback[t-1]
#             c = int(c_rw[t-1])
#
#             PE = f - c  # Prediction error
#             c_rw[t] = c + alpha*PE  # Update rule
#
#             # Ensure c_rw is between 0 and 100.
#             c_rw[t] = max(0, min(100, c_rw[t]))
#
#             # Get performance delta
#             delta_p = performance[t] - performance[t-1]
#
#             # Encure delta_p is between -100 and 100.
#             delta_p = max(-100, min(100, delta_p))
#
#             model_pred[t] = max(0, min(100, intercept + int((w_RW*c_rw[t]) + (w_PD*delta_p))))
#
#             # Calculate probabilities across different options (p_options)
#             conf_sim, p_options = sim_norm_prob_vectorized(
#                                                      np.array([model_pred[t]]),
#                                                      np.array([sigma]),
#                                                      lower_bound=0,
#                                                      upper_bound=100)
#             conf_vec[t] = conf_sim[0]
#
#     return conf_vec
#
# def RWPD_trial (x, *args):
#
#     """
#     Confidence is updated as the weighted sum of: (1) confidence updated by a
#     rescorla-wagner updated rule (C_RW) and (2) the change in performance
#     since the last trial (delta P).
#
#     return list of vectors
#     """
#
#     alpha, sigma, bias, w_RW, w_PD, intercept = x
#     confidence, feedback, n_trials, performance = args
#
#     sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
#     model_pred = np.zeros(n_trials)
#     c_rw = np.zeros(n_trials)
#     # Small value to avoid division by 0
#     epsilon = 1e-10
#     for t in range(n_trials):
#
#         if t == 0:
#             # The probability mean for the first trial (bias) is fit
#             # Trial included to get previous feedback and performance
#             model_pred[t] = bias
#             c_rw[t] = bias
#
#         else:
#
#             # Get previous confidence value and feedback,
#             f = feedback[t-1]
#             c = int(c_rw[t-1])
#
#             PE = f - c  # Prediction error
#             c_rw[t] = c + alpha*PE  # Update rule
#
#             # Encure c_rw is between 0 and 100.
#             c_rw[t] = max(0, min(100, c_rw[t]))
#
#             # Get delta performance
#             delta_p = performance[t] - performance[t-1]
#
#             # Encure delta_p is between -100 and 100.
#             delta_p = max(-100, min(100, delta_p))
#
#             # Get the prediction as the weighted sum
#             model_pred[t] = max(0, min(100, intercept + int((w_RW*c_rw[t]) + (w_PD*delta_p))))
#
#     # Remove initial baseline trial
#     model_pred = model_pred[1:]
#     sigma_vec = sigma_vec[1:]
#     confidence = confidence[1:]
#
#     # Calculate probabilities and Negative log likelihood (NLL)
#     probabilities, y_values = calc_norm_prob_vectorized(confidence,
#                                                         model_pred,
#                                                         sigma_vec)
#     nlls = -np.log(probabilities + epsilon)
#
#     return [nlls, model_pred, sigma_vec, confidence]
#
# def RWPD (x, *args):
#
#     """
#     Confidence is updated as the weighted sum of: (1)
#     confidence updated by a rescorla-wagner updated rule (C_RW) and (2)
#     the change in performance since the last trial (delta P).
#     """
#
#     alpha, sigma, bias, w_RW, w_PD, intercept = x
#     confidence, feedback, n_trials, performance, trial_index = args
#
#     sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
#     model_pred = np.zeros(n_trials)
#     c_rw = np.zeros(n_trials)
#     # Small value to avoid division by 0
#     epsilon = 1e-10
#     for t in range(n_trials):
#
#         if t == 0:
#             # The probability mean for the first trial (bias) is fit
#             # Trial included to get previous feedback and performance
#             model_pred[t] = bias
#             c_rw[t] = bias
#         else:
#             # Get previous feedback (f) and confidence estimate (c)
#             f = feedback[t-1]
#             c = int(c_rw[t-1])
#
#             PE = f - c  # Prediction error
#             c_rw[t] = c + alpha*PE  # Update rule
#
#             # Ensure c_rw is between 0 and 100.
#             c_rw[t] = max(0, min(100, c_rw[t]))
#
#             # Get performance delta
#             delta_p = performance[t] - performance[t-1]
#
#             # Encure delta_p is between -100 and 100.
#             delta_p = max(-100, min(100, delta_p))
#
#             model_pred[t] = max(0, min(100, intercept + int((w_RW*c_rw[t]) + (w_PD*delta_p))))
#
#     # Remove initial baseline trial
#     model_pred = model_pred[1:]
#     sigma_vec = sigma_vec[1:]
#     confidence = confidence[1:]
#
#     # Calculate probabilities and Negative log likelihood (NLL)
#     probabilities, y_values = calc_norm_prob_vectorized(confidence,
#                                                         model_pred,
#                                                         sigma_vec)
#     nlls = -np.log(probabilities + epsilon)
#
#     return np.sum(nlls[trial_index])
# =============================================================================

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

            # Update RW estimate  --------------------------------------

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

            # Update choice kernels -------------------------------------

            for k in range(101):

                # Update rule
                a_k_t = 1 if k == c else 0
                choice_kernels[k] += alpha_ck * (a_k_t - choice_kernels[k])

            # Gaussian smooting
            smoothed_choice_kernels = gaussian_filter1d(choice_kernels,
                                                        sigma_ck)

            # Combine RW update and CK update ---------------------------

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
    conf_dist_vec = np.zeros((n_trials,101))
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
            conf_dist_vec[t] = y_values

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
            conf_dist_vec[t] = y_values

    return conf_vec, conf_dist_vec

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
