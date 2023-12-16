# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:26:08 2023

@author: carll
"""

# Models
import numpy as np
from scipy.optimize import minimize
import math
from src.utility_functions import (create_probability_distribution,
                                   invert_normal_distribution,
                                   create_truncated_normal)


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

        if t == 0:  # Baseline trial, only included to index passed c and f.
            continue

        if t == 1:
            # The probability mean for the first trial (bias) is fit
            model_pred[t] = bias

        if t > 1:

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

    # Remove initial baseline trial
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

    alpha, sigma, bias = x
    confidence, feedback, n_trials = args

    sigma_vec = np.full(n_trials, sigma)  # vector for standard deviation
    model_pred = np.zeros(n_trials)
    for t in range(n_trials):

        if t == 0:  # Baseline trial, only included to index passed c and f.
            continue

        if t == 1:
            # The probability mean for the first trial (bias) is fit
            model_pred[t] = bias

        if t > 1:

            # Get previous confidence value and feedback,
            f = feedback[t-1]
            c = int(confidence[t-1])

            PE = f - c  # Prediction error
            c_pred = c + alpha*PE  # Update rule

            # Encure c_pred is between 1 and 100.
            c_pred = max(0, min(100, c_pred))
            model_pred[t] = c_pred

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
        if t == 0:  # Baseline trial, only included to index passed c and f.
            continue

        if t == 1:
            # Keep to previous confidence
            c = int(confidence[t-1])

            # Get option probabilities
            prob_distribution = create_probability_distribution(options,
                                                                c,
                                                                sigma)

            prob_dists.append(prob_distribution)

            # Get negative log likelihood of reported confidence
            log_likelihood = np.log(prob_distribution[confidence[t]]
                                    + epsilon)
            nll_vector[t] = -log_likelihood

        if t > 1:

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
        if t == 0:  # Baseline trial, only included to index passed c and f.
            continue

        if t == 1:
            # Keep to previous confidence
            c = int(confidence[t-1])

            # Get option probabilities
            prob_distribution = create_probability_distribution(options,
                                                                c,
                                                                sigma)
            # Get negative log likelihood of confidence
            log_likelihood = np.log(prob_distribution[int(confidence[t])]
                                    + epsilon)
            nll_vector[t] = -log_likelihood

        else:
            # Get previous confidence value and set
            f = feedback[t-1]
            c = int(confidence[t-1])-1

            # Set win boundary
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary

            if f > lower_bound and f < upper_bound:  # win-trial

                # Get option probabilities
                prob_distribution = create_probability_distribution(options,
                                                                    c,
                                                                    sigma)

            else:  # lose-trial

                # Get probability function
                prob_dist = create_probability_distribution(options,
                                                            c,
                                                            sigma)

                # inverted distribution
                prob_distribution = invert_normal_distribution(prob_dist)

            # Get negative log likelihood of confidence
            log_likelihood = np.log(prob_distribution[int(confidence[t])]
                                    + epsilon)
            nll_vector[t] = -log_likelihood

    # remove baseline
    nll_vector = nll_vector[1:]

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
        if t == 0:  # Baseline trial, nll not included for this trial
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

        if t == 0:  # Baseline trial, only included to index passed c and f.
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

    # remove baseline
    nll_vector = nll_vector[1:]

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
    n_trials = n_trials-1  # Remove baseline session
    random_choice_probability = 1/prediction_range
    log_likelihood = math.log(random_choice_probability)
    nll_vector = [-log_likelihood]*n_trials

    return np.sum(nll_vector)


def fit_model(model, args, bounds, n_trials, start_value_number=10,
              solver="L-BFGS-B"):
    """
    Fits a model using the minimize function with various start values.

    This function optimizes a model's parameters by exploring multiple
    starting points within specified bounds. It uses scipy.optimize's
    minimize function and selects the best result based on the lowest
    NLL (negative log likelihood). It also calculates additional
    statistical measures like Akaike Information Criterion (AIC),
    Bayesian Information Criterion (BIC), and McFadden's pseudo
    R-squared.

    Parameters:
    model (callable): The model function to optimize.
    args (tuple): Arguments to pass to the model function.
    bounds (list of tuples): Parameter bounds in the format (min, max).
    n_trials (int): Number of trials in the dataset.
    start_value_number (int, optional): Number of start values for each
                                        parameter. Default is 10.
    solver (str, optional): Optimization algorithm. Default is "L-BFGS-B".

    Returns:
    list: Best-fitting parameters, NLL, AIC, BIC, and McFadden's
          pseudo R-squared.
    """

    # Get start ranges for parameters
    start_ranges = []
    for bound in bounds:
        lower_bound, upper_bound = bound
        start_range = np.random.uniform(lower_bound,
                                        upper_bound,
                                        size=start_value_number)
        start_ranges.append(start_range)

    # Loop over start values
    result_list = []
    nll_list = []
    for start_values in zip(*start_ranges):

        result = minimize(model,
                          start_values,
                          args=args,
                          bounds=bounds,
                          method=solver)
        # Get results
        best_params = result.x
        nll = result.fun

        # McFadden pseudo r2
        nll_model = nll
        n = n_trials
        nll_null = -n * np.log(0.5)
        pseudo_r2 = 1 - (nll_model / nll_null)

        # Get AIC and BIC
        k = len(start_values)  # Number of parameters fitted
        aic = 2*k + 2*nll
        bic = k*np.log(n) + 2*nll

        # Append to result and nll lists
        result_list.append(list(best_params) + [nll, aic, bic, pseudo_r2])
        nll_list.append(nll)

    # Return best result
    best_result_index = nll_list.index(min(nll_list))
    best_result = result_list[best_result_index]

    return best_result
