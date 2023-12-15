# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:32:41 2023

@author: carll
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm

# Utility functions


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


def create_truncated_normal(mean, sd, lower_bound=0, upper_bound=100):
    """
    Creates a truncated normal distribution within specified bounds.

    This function generates a truncated normal distribution with a given mean
    and standard deviation, truncated within specified lower and upper bounds.
    The distribution is truncated so that values lie within the given range,
    ensuring the integral of the distribution's probability density function
    is 1 within these bounds.

    Parameters:
    mean (float): The mean of the normal distribution.
    sd (float): The standard deviation of the normal distribution.
    lower_bound (float, optional): The lower boundary for truncation of the
                                   normal distribution. Default is 0.
    upper_bound (float, optional): The upper boundary for truncation of the
                                   normal distribution. Default is 100.

    Returns:
    scipy.stats.rv_continuous: A truncated normal continuous random variable
                               object as defined in scipy.stats.

    Note:
    The function adjusts the bounds to the standard normal distribution's
    z-scores before creating the truncated distribution. This distribution
    can be used to generate random variates and to compute various statistics.
    """

    # Convert the bounds to z-scores
    lower_bound_z = (lower_bound - mean) / sd
    upper_bound_z = (upper_bound - mean) / sd

    # Create the truncated normal distribution
    return truncnorm(lower_bound_z, upper_bound_z, loc=mean, scale=sd)


def probs_from_trunc_norm(mean, sd, lower_bound=0, upper_bound=100, c=None):
    """
    Generate probabilities from a truncated normal distribution. This
    function defines the distribution by a mean, standard deviation, and
    truncates it between given lower and upper bounds. It's useful for
    probability distributions within a specific range.

    :param mean: Mean of the normal distribution.
    :param sd: Standard deviation of the normal distribution.
    :param lower_bound: Lower bound for truncation (default 0).
    :param upper_bound: Upper bound for truncation (default 100).
    :param c: Optional specific choice for probability calculation
              (default None).
    :return: A tuple with the probability of choice 'c' (or None) and
             an array of normalized probabilities for each option.

    The function converts bounds to z-scores, creates a truncated
    normal distribution, and computes normalized probabilities.
    """

    # Convert the bounds to z-scores
    lower_bound_z = (lower_bound - mean) / sd
    upper_bound_z = (upper_bound - mean) / sd

    # Create the truncated normal distribution
    trunc_normal = truncnorm(lower_bound_z, upper_bound_z, loc=mean, scale=sd)

    # Option probabilities
    option_probs = trunc_normal.pdf(np.linspace(1, 100, 100))

    # Normalize the probabilities so they sum to 1
    option_probs_normalised = option_probs / np.sum(option_probs)

    # Get choice probability for a specific choice 'c', if 'c' is provided
    choice_prob = option_probs_normalised[c - 1] if c is not None else None

    return choice_prob, option_probs_normalised


def invert_continuous_distribution(pdf, x_range, mean):
    """
    Inverts and normalizes a continuous probability density function (pdf).
    The function takes a pdf and inverts it by subtracting its values from
    the maximum probability density in the specified range. After inversion,
    it normalizes the distribution to ensure the total area under the curve
    equals 1.

    Parameters:
    pdf (callable): The continuous pdf to be inverted.
    x_range (tuple): The range (start, end) over which the distribution
                     is defined.

    Returns:
    callable: A normalized, inverted probability density function.
    """

    # Find the maximum probability density in the given range
    max_pdf = pdf(mean)

    # Define the inverted pdf
    def inverted_pdf(x):
        return max_pdf - pdf(x)

    # Normalize the inverted pdf
    normalization_factor, _ = quad(inverted_pdf, x_range[0], x_range[1])

    def normalized_inverted_pdf(x):
        return inverted_pdf(x) / normalization_factor

    return normalized_inverted_pdf


def invert_normal_distribution(probabilities):

    """
    Inverts a probability distribution and normalizes the sum to 1.

    This function inverts each probability value in a given probability
    distribution relative to the maximum probability, then normalizes the
    resulting values. The sum of the inverted distribution is maintained at 1.
    This process reflects the distribution around its maximum value, useful
    for creating complementary probability distributions.

    Parameters:
    probabilities (array-like): An array representing a probability
                                distribution.

    Returns:
    array-like: An array representing the inverted and normalized
                distribution.

    Note:
    Inversion is relative to the maximum value in the input distribution.
    For a distribution with a highest probability P, each probability p is
    inverted as (P - p). The values are then normalized to ensure their sum
    is 1, retaining the characteristics of a probability distribution.
    """

    # Find the maximum probability
    max_prob = np.max(probabilities)
    # Invert the probabilities
    inverted = max_prob - probabilities
    # Normalize the inverted probabilities
    normalized_inverted = inverted / np.sum(inverted)

    return normalized_inverted


def create_probability_distribution(options=[1, 2, 3],
                                    mean_option=50,
                                    sigma=10):
    """
    Create a normalized probability distribution using a normal (Gaussian)
    distribution for a specified set of options. This distribution is
    centered around a chosen option (mean_option) with a designated
    standard deviation (sigma). It's ideal for scenarios where the
    likelihood of certain options varies based on their closeness to
    mean_option.

    Parameters:
    options: A list or array of options to calculate probabilities for
             (default [1,2,3]).
    mean_option: The central value (mean) of the normal distribution
                 (default 50).
    sigma: Standard deviation, determining the distribution's spread
           (default 10).
    Returns:
    An array of probabilities for each option, reflecting the normalized
    normal distribution for these options.
    """

    # Creating a normal distribution around the mean with normalization factor
    factor = 1 / (sigma * np.sqrt(2 * np.pi))
    probabilities = factor * np.exp(-0.5 * ((options - mean_option) / sigma)
                                    ** 2)

    # Normalize the distribution so that the sum of probabilities equals 1
    probabilities /= probabilities.sum()

    return probabilities

