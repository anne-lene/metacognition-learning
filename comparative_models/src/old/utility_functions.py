# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:32:41 2023

@author: carll
"""

import numpy as np
from scipy.integrate import quad
from scipy.stats import truncnorm
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch
from matplotlib.path import Path
import pandas as pd
import os
from tqdm import tqdm
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

class HandlerNormalDist(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Scaling factors for the size of the curve
        scale_x = width / 4
        scale_y = height * 1

        # Horizontal shift to move the curve to the right
        shift_x = width / 1.5 # Adjust this value to shift the curve

        # Create a path for a normal distribution shape
        x = np.linspace(-2, 2, 100)
        y = np.exp(-x**2)
        vertices = [(x[i] * scale_x + shift_x, y[i] * scale_y) for i in range(len(x))]
        vertices = [(-2 * scale_x + shift_x, 0)] + vertices + [(2 * scale_x + shift_x, 0)]
        codes = [Path.MOVETO] + [Path.LINETO] * len(x) + [Path.LINETO]
        path = Path(vertices, codes)

        # Create a patch from this path
        patch = mpatches.PathPatch(path, #facecolor='none',
                                   edgecolor=orig_handle.get_edgecolor(),
                                   lw=0,
                                   )
        patch.set_transform(trans)
        return [patch]


def write_metrics_to_excel(filename, **kwargs):
    # Create a Pandas Excel writer using XlsxWriter as the engine
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        for sheet_name, data in kwargs.items():
            # Convert the data (assumed to be a numpy array) into a DataFrame
            df = pd.DataFrame(data)
            # Write the DataFrame to a specific Excel sheet
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def read_metrics_from_excel(filename):
    metrics_dict = {}
    with pd.ExcelFile(filename) as xls:
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name)
            metrics_dict[sheet_name] = df.to_numpy().squeeze()
    return metrics_dict

# Function to automatically assign metrics from a dictionary to global variables
def assign_metrics_from_dict(metrics_dict):
    for variable_name, data_array in metrics_dict.items():
        globals()[variable_name] = data_array

def load_fixed_feedback_data():

    # Import data
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    grandparent_directory = os.path.dirname(parent_directory)
    project_path = grandparent_directory
    experiment_path = r"fixed_feedback"
    fixed_feedback_data_path = r'data/cleaned'
    data_file = r'main-20-12-14-processed_filtered.csv'
    full_path = os.path.join(project_path, experiment_path,
                             fixed_feedback_data_path, data_file)
    df = pd.read_csv(full_path, low_memory=False)

    # Add session column
    df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

    error_baseline_list = []
    performance_baseline_list  = []
    error_task_list  = []
    performance_task_list  = []
    bdi_list  = []
    condition_list = []
    pid_list = []
    pid_nr_list = []
    confidence_baseline_list = []
    confidence_task_list = []
    p_avg_task_list = []
    p_avg_baseline_list = []
    error_change_baseline_list = []
    error_change_task_list = []
    estimate_baseline_list = []
    estimate_task_list = []
    subtrial_variance_baseline = []
    subtrial_std_baseline = []
    subtrial_variance_task = []
    subtrial_std_task = []
    a_empirical_list = []

    subtrial_error_mean_baseline_list = []
    subtrial_error_variance_baseline_list = []
    subtrial_error_std_baseline_list = []
    subtrial_error_mean_task_list = []
    subtrial_error_variance_task_list = []
    subtrial_error_std_task_list = []

    feedback_list = []
    session_list = []

    # Loop over participants
    for pid_nr, participant in enumerate(tqdm(df.pid.unique(),
                                         total=len(df.pid.unique()),
                                         desc='Participant loop')):

        df_p = df[df.pid==participant]

        error_baseline_p = []
        performance_baseline_p = []

        error_task_p = []
        performance_task_p = []

        bdi_p = []
        condition_p = []
        pid_p = []
        pid_nr_p = []



        # Loop over sessions
        for session in df_p.session.unique():

            # Get current session data
            df_s = df_p[df_p.session == session].copy()

            # get the mean, std, and var of estimate of across all subtrials
            df_s['mean_estimate'] = df_s.groupby('trial')['estimate'].transform('mean')
            df_s['std_estimate'] = df_s.groupby('trial')['estimate'].transform('std')
            df_s['var_estimate'] = df_s.groupby('trial')['estimate'].transform('var')

            # Calculate the absolute error for each subtrial
            df_s['abs_error'] = (df_s['estimate'] - df_s['correct']).abs()

            # Get the mean, variance and std of absolute error
            df_s['mean_error'] = df_s.groupby('trial')['abs_error'].transform('mean')
            df_s['std_error'] = df_s.groupby('trial')['abs_error'].transform('std')
            df_s['var_error'] = df_s.groupby('trial')['abs_error'].transform('var')

            # Only consider the last subtrial
            df_s = df_s.drop_duplicates(subset='trial', keep='last')

            # Filter out all 'baseline' rows
            df_s_task = df_s[df_s['condition'] != 'baseline']

            # Only take the last 10 trils in baseline
            df_s_baseline = df_s[df_s['condition'] == 'baseline'].tail(10)

            # Get values
            n_trials = len(df_s_task)
            condition = df_s_task.condition.unique()[0]
            confidence = df_s_task.confidence.values
            feedback = df_s_task.feedback.values

            # Depression measure
            bdi = df_s.bdi.values[0]

            # dot estimates
            estimate_baseline = df_s_baseline.mean_estimate.values
            estimate_task = df_s_task.mean_estimate.values

            estimate_std_baseline = df_s_baseline.std_estimate.values
            estimate_std_task = df_s_task.std_estimate.values

            estimate_var_baseline = df_s_baseline.var_estimate.values
            estimate_var_task = df_s_task.var_estimate.values

            # Error stats from all subtrials
            subtrial_error_mean_baseline = df_s_baseline.mean_error.values
            subtrial_error_std_baseline = df_s_baseline.std_error.values
            subtrial_error_var_baseline = df_s_baseline.var_error.values

            subtrial_error_mean_task = df_s_task.mean_error.values
            subtrial_error_std_task = df_s_task.std_error.values
            subtrial_error_var_task = df_s_task.var_error.values

            # Error and performance - brased on last subtrial
            error_baseline = df_s_baseline.estimate.values - df_s_baseline.correct.values
            performance_baseline = abs(error_baseline)

            error_task = df_s_task.estimate.values - df_s_task.correct.values
            performance_task = abs(error_task)

            p_avg_task = df_s_task.pavg.values
            p_avg_baseline = df_s_baseline.pavg.values

            # Confidence
            confidence_task = df_s_task.confidence.values
            confidence_baseline = df_s_baseline.confidence.values


            # Empirical learning rate
            c_t1_vector = df_s_task.confidence.values[1:]  # Shift forward
            c_t_vector = df_s_task.confidence.values[:-1] # remove last step
            fb_vector =  df_s_task.feedback.values[:-1]  # remove last step
            epsilon = 1e-10 # Prevent division by 0
            a_empirical =  [((c_t1-c_t)/((fb-c_t)+epsilon))
                            for c_t1, c_t, fb in zip(
                            c_t1_vector, c_t_vector, fb_vector)]

            # Trial-to-Trial change in error
            error_change_baseline = np.diff(performance_baseline)
            error_change_task = np.diff(performance_task)

            # Save to list
            subtrial_variance_baseline.append(estimate_var_baseline)
            subtrial_std_baseline.append(estimate_std_baseline )
            subtrial_variance_task.append(estimate_var_task)
            subtrial_std_task.append(estimate_std_task)

            subtrial_error_mean_baseline_list.append(subtrial_error_mean_baseline)
            subtrial_error_variance_baseline_list.append(subtrial_error_var_baseline)
            subtrial_error_std_baseline_list.append(subtrial_error_std_baseline)
            subtrial_error_mean_task_list.append(subtrial_error_mean_task)
            subtrial_error_variance_task_list.append(subtrial_error_var_task)
            subtrial_error_std_task_list.append(subtrial_error_std_task)

            error_baseline_list.append(error_baseline)
            performance_baseline_list.append(performance_baseline)
            error_task_list.append(error_task)
            performance_task_list.append(performance_task)
            bdi_list.append(bdi)
            condition_list.append(condition)
            pid_list.append(participant)
            pid_nr_list.append(pid_nr)
            p_avg_task_list.append(p_avg_task)
            p_avg_baseline_list.append(p_avg_baseline)
            error_change_baseline_list.append(error_change_baseline)
            error_change_task_list.append(error_change_task)
            confidence_baseline_list.append(confidence_baseline)
            confidence_task_list.append(confidence_task)
            estimate_baseline_list.append(estimate_baseline)
            estimate_task_list.append(estimate_task)
            a_empirical_list.append(a_empirical)
            feedback_list.append(feedback)
            session_list.append(session)

    df_a = pd.DataFrame({'error_baseline': error_baseline_list,
                         'performance_baseline': performance_baseline_list,
                         'error_task': error_task_list,
                         'performance_task': performance_task_list,
                         'bdi': bdi_list,
                         'condition_list': condition_list,
                         'pid': pid_list,
                         'pid_nr': pid_nr_list,
                         'confidence_baseline': confidence_baseline_list,
                         'confidence_task': confidence_task_list,
                         'p_avg_task': p_avg_task_list,
                         'p_avg_baseline': p_avg_baseline_list,
                         'error_change_baseline':error_change_baseline_list,
                         'error_change_task': error_change_task_list,
                         'estimate_baseline': estimate_baseline_list,
                         'estimate_task': estimate_task_list,
                         'subtrial_std_baseline': subtrial_std_baseline,
                         'subtrial_var_baseline': subtrial_variance_baseline,
                         'subtrial_std_task': subtrial_std_task,
                         'subtrial_var_task': subtrial_variance_task,
                         'subtrial_error_mean_baseline': subtrial_error_mean_baseline_list,
                         'subtrial_error_variance_baseline': subtrial_error_variance_baseline_list,
                         'subtrial_error_std_baseline': subtrial_error_std_baseline_list,
                         'subtrial_error_mean_task': subtrial_error_mean_task_list,
                         'subtrial_error_variance_task': subtrial_error_variance_task_list,
                         'subtrial_error_std_task': subtrial_error_std_task_list,
                         'a_empirical': a_empirical_list,
                         'feedback': feedback_list,
                         'session': session_list
                         })

    return df_a







