# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:45:44 2025

@author: carll
"""

# Utility functions
import os
import pandas as pd
import numpy as np

def load_df(EXP):
    """
    Load a CSV file using a relative path based on the script's location.

    Parameters:
        relative_path (str or Path): Relative path to the CSV file from the
        directory of this script.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path
    if EXP == 1:
        relative_path = "../data/EXP1_fixed_feedback/main-20-12-14-processed_filtered.csv"
    elif EXP == 2:
        relative_path = "../data/EXP2_varied_feedback/variable_fb_data_full_processed.csv"
    else:
        raise ValueError("EXP needs to be 1 or 2")

    # Construct the full path to the CSV file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    return pd.read_csv(file_path, low_memory=False)

def load_sim_df(EXP, custom_relative_path=False):
    """
    Load a CSV file using a relative path based on the script's location.

    Parameters:
        relative_path (str or Path): Relative path to the CSV file from the
        directory of this script.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path
    if custom_relative_path:
        relative_path = custom_relative_path
    else:
        if EXP == 1:
            relative_path = "../results/Fixed_feedback/model_comparison/model_and_param_recovery/model_simulations_EXP1.csv"
        elif EXP == 2:
            relative_path = "../results/variable_feedback/model_comparison/model_and_param_recovery/model_simulations_EXP2.csv"
        else:
            raise ValueError("EXP needs to be 1 or 2")

    # Construct the full path to the CSV file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    return pd.read_csv(file_path, low_memory=False)

def load_fit_on_sim_df(EXP, custom_relative_path=False):
    """
    Load a CSV file using a relative path based on the script's location.

    Parameters:
        relative_path (str or Path): Relative path to the CSV file from the
        directory of this script.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path
    if custom_relative_path:
        relative_path = custom_relative_path
    else:
        if EXP == 1:
            relative_path = "../results/Fixed_feedback/model_comparison/model_and_param_recovery/model_fits_EXP1_sim.xlsx"
        elif EXP == 2:
            relative_path = "../results/variable_feedback/model_comparison/model_and_param_recovery/model_fits_EXP2_sim.xlsx"
        else:
            raise ValueError("EXP needs to be 1 or 2")

    # Construct the full path to the CSV file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    return pd.read_excel(file_path)

def load_fit_on_data_df(EXP, custom_relative_path=False):
    """
    Load a CSV file using a relative path based on the script's location.

    Parameters:
        relative_path (str or Path): Relative path to the CSV file from the
        directory of this script.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path
    if custom_relative_path:
        relative_path = custom_relative_path
    else:
        if EXP == 1:
            relative_path = "../results/Fixed_feedback/model_comparison/models_fit_to_data/model_fits_EXP1_data.xlsx"
        elif EXP == 2:
            relative_path = "../results/variable_feedback/model_comparison/models_fit_to_data/model_fits_EXP2_data.xlsx"
        else:
            raise ValueError("EXP needs to be 1 or 2")

    # Construct the full path to the CSV file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))

    return pd.read_excel(file_path)

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
