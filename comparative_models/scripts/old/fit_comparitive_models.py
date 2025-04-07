# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:02:33 2023

@author: carll
"""

# Analysing data - Fixed feedback condition

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from src.utility_functions import add_session_column
from src.models import (fit_model,
                        fit_model_with_cv,
                        fit_random_model,
                        random_model,
                        random_model_w_bias,
                        win_stay_lose_shift,
                        rw_symmetric_LR,
                        choice_kernel,
                        RW_choice_kernel,
                        delta_P_RW)

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)

df = pd.read_csv(full_path, low_memory=False)
#%%
# Add session column
df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Number of participants and maximum number of sessions per participant
num_participants = len(df.pid.unique())
#max_sessions = df.groupby('pid')['session'].nunique().max()

results = []


# Pre-allocate arrays for participant-level metrics
# Metrics for Random Model
nll_array_random_p = np.zeros(num_participants)
aic_array_random_p = np.zeros(num_participants)
bic_array_random_p = np.zeros(num_participants)
pseudo_r2_array_random_p = np.zeros(num_participants)

# Metrics for Bias Model
mean_array_bias_p = np.zeros(num_participants)
sd_array_bias_p = np.zeros(num_participants)
nll_array_bias_p = np.zeros(num_participants)
aic_array_bias_p = np.zeros(num_participants)
bic_array_bias_p = np.zeros(num_participants)
pseudo_r2_array_bias_p = np.zeros(num_participants)

# Metrics for Win-Stay-Lose-Shift Model
std_WSLS_array_p = np.zeros(num_participants)
win_boundary_WSLS_array_p = np.zeros(num_participants)
nll_array_win_stay_p = np.zeros(num_participants)
aic_array_win_stay_p = np.zeros(num_participants)
bic_array_win_stay_p = np.zeros(num_participants)
pseudo_r2_array_win_stay_p = np.zeros(num_participants)

# Metrics for RW Symmetric Model
alpha_array_rw_symm_p = np.zeros(num_participants)
sigma_array_rw_symm_p = np.zeros(num_participants)
bias_array_rw_symm_p = np.zeros(num_participants)
nll_array_rw_symm_p = np.zeros(num_participants)
aic_array_rw_symm_p = np.zeros(num_participants)
bic_array_rw_symm_p = np.zeros(num_participants)
pseudo_r2_array_rw_symm_p = np.zeros(num_participants)

# Metrics for choice kernel
alpha_array_ck_p = np.zeros(num_participants)
sigma_array_ck_p = np.zeros(num_participants)
bias_array_ck_p = np.zeros(num_participants)
beta_array_ck_p = np.zeros(num_participants)
nll_array_ck_p = np.zeros(num_participants)
aic_array_ck_p = np.zeros(num_participants)
bic_array_ck_p = np.zeros(num_participants)
pseudo_r2_array_ck_p = np.zeros(num_participants)

# Metrics for RW + choice kernel
alpha_array_rwck_p = np.zeros(num_participants)
alpha_ck_array_rwck_p = np.zeros(num_participants)
sigma_array_rwck_p = np.zeros(num_participants)
bias_array_rwck_p = np.zeros(num_participants)
beta_array_rwck_p = np.zeros(num_participants)
beta_ck_array_rwck_p = np.zeros(num_participants)
nll_array_rwck_p = np.zeros(num_participants)
aic_array_rwck_p = np.zeros(num_participants)
bic_array_rwck_p = np.zeros(num_participants)
pseudo_r2_array_rwck_p = np.zeros(num_participants)

# Metrics for Delta P RW Model
alpha_array_delta_p_rw_p = np.zeros(num_participants)
sigma_array_delta_p_rw_p = np.zeros(num_participants)
bias_array_delta_p_rw_p = np.zeros(num_participants)
w_rw_array_delta_p_rw_p = np.zeros(num_participants)
w_delta_p_array_delta_p_rw_p = np.zeros(num_participants)
nll_array_delta_p_rw_p = np.zeros(num_participants)
aic_array_delta_p_rw_p = np.zeros(num_participants)
bic_array_delta_p_rw_p = np.zeros(num_participants)
pseudo_r2_array_delta_p_rw_p = np.zeros(num_participants)

# Participant index
participant_idx = 0

# Loop over participants
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):

    # Current participant data only
    df_p = df[df['pid'] == participant]

    # Numer of sessions for this participant
    max_sessions = df_p['session'].nunique()

    # Arrays for Random Model
    nll_array_random = np.zeros(max_sessions)
    aic_array_random = np.zeros(max_sessions)
    bic_array_random = np.zeros(max_sessions)
    pseudo_r2_array_random = np.zeros(max_sessions)

    # Arrays for Bias Model
    mean_array_bias = np.zeros(max_sessions)
    sd_array_bias = np.zeros(max_sessions)
    nll_array_bias = np.zeros(max_sessions)
    aic_array_bias = np.zeros(max_sessions)
    bic_array_bias = np.zeros(max_sessions)
    pseudo_r2_array_bias = np.zeros(max_sessions)

    # Arrays for Win-Stay-Lose-Shift Model
    std_WSLS_array = np.zeros(max_sessions)
    win_boundary_WSLS_array = np.zeros(max_sessions)
    nll_array_win_stay = np.zeros(max_sessions)
    aic_array_win_stay = np.zeros(max_sessions)
    bic_array_win_stay = np.zeros(max_sessions)
    pseudo_r2_array_win_stay = np.zeros(max_sessions)

    # Arrays for RW Symmetric Model
    alpha_array_rw_symm = np.zeros(max_sessions)
    sigma_array_rw_symm = np.zeros(max_sessions)
    bias_array_rw_symm = np.zeros(max_sessions)
    nll_array_rw_symm = np.zeros(max_sessions)
    aic_array_rw_symm = np.zeros(max_sessions)
    bic_array_rw_symm = np.zeros(max_sessions)
    pseudo_r2_array_rw_symm = np.zeros(max_sessions)

    # Arrays for choice kernel model
    alpha_array_ck = np.zeros(max_sessions)
    sigma_array_ck = np.zeros(max_sessions)
    bias_array_ck = np.zeros(max_sessions)
    beta_array_ck = np.zeros(max_sessions)
    nll_array_ck = np.zeros(max_sessions)
    aic_array_ck = np.zeros(max_sessions)
    bic_array_ck = np.zeros(max_sessions)
    pseudo_r2_array_ck = np.zeros(max_sessions)

    # Arrays for RW + choice kernel model
    alpha_array_rwck = np.zeros(max_sessions)
    alpha_ck_array_rwck = np.zeros(max_sessions)
    sigma_array_rwck = np.zeros(max_sessions)
    bias_array_rwck = np.zeros(max_sessions)
    beta_array_rwck = np.zeros(max_sessions)
    beta_ck_array_rwck = np.zeros(max_sessions)
    nll_array_rwck = np.zeros(max_sessions)
    aic_array_rwck = np.zeros(max_sessions)
    bic_array_rwck = np.zeros(max_sessions)
    pseudo_r2_array_rwck = np.zeros(max_sessions)

    # Arrays for Delta P RW Model
    alpha_array_delta_p_rw = np.zeros(max_sessions)
    sigma_array_delta_p_rw = np.zeros(max_sessions)
    bias_array_delta_p_rw = np.zeros(max_sessions)
    w_rw_array_delta_p_rw = np.zeros(max_sessions)
    w_delta_p_array_delta_p_rw = np.zeros(max_sessions)
    nll_array_delta_p_rw = np.zeros(max_sessions)
    aic_array_delta_p_rw = np.zeros(max_sessions)
    bic_array_delta_p_rw = np.zeros(max_sessions)
    pseudo_r2_array_delta_p_rw = np.zeros(max_sessions)

    # Session index
    session_idx = 0

    # Loop over sessions
    for session in df_p.session.unique():

        # Get current session data, one row per trial
        df_s = df_p[df_p.session == session]

        # Only feedback trials
        df_s = df_s[df_s.condition != 'baseline']

        # Calculate the difference between 'estimate' and 'correct'
        df_s['difference'] = df_s['estimate'] - df_s['correct']

        # Group by 'subtrial' and calculate the mean of the differences
        error_avg = df_s.groupby('trial')['difference'].mean()

        # One row per trial
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        # Get variables
        confidence = df_s.confidence.values
        feedback = df_s.feedback.values
        n_trials = len(df_s)
        performance = -error_avg.values # df_s.estimate.values - df_s.correct.values #df_s.pavg.values

        # Random confidence model
        bounds = [(0, 0)] # No parameter being fitted.
        results = fit_model_with_cv(model=random_model,
                                    args=(100, # Prediction_range
                                          n_trials),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

        x = results[0] # Non-existing param only included to run same procedure as other models.
        nll_array_random[session_idx] = results[1]
        aic_array_random[session_idx] = results[2]
        bic_array_random[session_idx] = results[3]
        pseudo_r2_array_random[session_idx] = results[4]

        # Biased confidence model
        # Set Bounds
        mean_bound = (0, 100)   # Mean
        sigma_bound = (1, 100)  # Standard deviation
        bounds = [(mean_bound[0], mean_bound[1]),
                  (sigma_bound[0],  sigma_bound[1]),
                  ]

        # Get results
        results = fit_model_with_cv(model=random_model_w_bias,
                            args=(confidence,
                                  n_trials),
                            bounds=bounds,
                            n_trials=n_trials,
                            start_value_number=50,
                            solver="L-BFGS-B")

        mean_array_bias[session_idx] = results[0]
        sd_array_bias[session_idx] = results[1]
        nll_array_bias[session_idx] = results[2]
        aic_array_bias[session_idx] = results[3]
        bic_array_bias[session_idx] = results[4]
        pseudo_r2_array_bias[session_idx] = results[5]

        # Win-stay-lose-shift-model (WSLS)
        # Set Bounds
        sigma_bound = (1, 100)  # Standard deviation
        win_bound = (1, 100)    # Win boundary
        bounds = [(sigma_bound[0], sigma_bound[1]),
                  (win_bound[0], win_bound[1])]

        # Get results
        results_win_stay = fit_model_with_cv(model=win_stay_lose_shift,
                                     args=(confidence,
                                           feedback,
                                           n_trials
                                           ),
                                     bounds=bounds,
                                     n_trials=n_trials,
                                     start_value_number=50,
                                     solver="L-BFGS-B")

        std_WSLS_array[session_idx] = results_win_stay[0]
        win_boundary_WSLS_array[session_idx] = results_win_stay[1]
        nll_array_win_stay[session_idx] = results_win_stay[2]
        aic_array_win_stay[session_idx] = results_win_stay[3]
        bic_array_win_stay[session_idx] = results_win_stay[4]
        pseudo_r2_array_win_stay[session_idx] = results_win_stay[5]

        # Rescorla-wagner model
        # Set bounds
        alpha_bound = (0, 1)    # Alpha
        sigma_bound = (1, 100)  # Standard deviation
        bias_bound = (0, 100)   # Mean at first trial
        bounds = [(alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1])]

        # Get results
        results_rw_symm = fit_model_with_cv(model=rw_symmetric_LR,
                                    args=(confidence,
                                          feedback,
                                          n_trials),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

        alpha_array_rw_symm[session_idx] = results_rw_symm[0]
        sigma_array_rw_symm[session_idx] = results_rw_symm[1]
        bias_array_rw_symm[session_idx] = results_rw_symm[2]
        nll_array_rw_symm[session_idx] = results_rw_symm[3]
        aic_array_rw_symm[session_idx] = results_rw_symm[4]
        bic_array_rw_symm[session_idx] = results_rw_symm[5]
        pseudo_r2_array_rw_symm[session_idx] = results_rw_symm[6]

        # Choice Kernel Model
        # Set bounds
        alpha_bound = (0, 1)  # Alpha previous (0.001, 1)
        sigma_bound = (1, 100)  # Standard deviation
        bias_bound = (0, 100)  # Mean at first trial
        beta_bound = (0, 5)  # Beta
        bounds = [(alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1]),
                  (beta_bound[0], beta_bound[1])]

        # Get results
        results_ck = fit_model_with_cv(model=choice_kernel,
                                    args=(confidence,
                                          n_trials,
                                          ),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

        alpha_array_ck[session_idx] = results_ck[0]
        sigma_array_ck[session_idx] = results_ck[1]
        bias_array_ck[session_idx] = results_ck[2]
        beta_array_ck[session_idx] = results_ck[3]
        nll_array_ck[session_idx] = results_ck[4]
        aic_array_ck[session_idx] = results_ck[5]
        bic_array_ck[session_idx] = results_ck[6]
        pseudo_r2_array_ck[session_idx] = results_ck[7]

        # RW + Choice Kernel Model
        # Set bounds
        alpha_bound = (0, 1)  # Alpha
        alpha_ck_bound = (0, 1)  # Alpha choice kernel
        sigma_bound = (1, 100)  # Standard deviation
        bias_bound = (0, 100)  # Mean at first trial
        beta_bound = (0, 5)  # Beta for RW
        beta_ck_bound = (0, 5)  # Beta for choice kernel
        bounds = [
                  (alpha_bound[0], alpha_bound[1]),
                  (alpha_ck_bound[0], alpha_ck_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1]),
                  (beta_bound[0], beta_bound[1]),
                  (beta_ck_bound[0], beta_ck_bound[1])
                  ]

        # Get results
        results_rwck = fit_model_with_cv(model=RW_choice_kernel,
                                 args=(feedback,
                                       confidence,
                                       n_trials,
                                       ),
                                 bounds=bounds,
                                 n_trials=n_trials,
                                 start_value_number=50,
                                 solver="L-BFGS-B")

        alpha_array_rwck[session_idx] = results_rwck[0]
        alpha_ck_array_rwck[session_idx] = results_rwck[1]
        sigma_array_rwck[session_idx] = results_rwck[2]
        bias_array_rwck[session_idx] = results_rwck[3]
        beta_array_rwck[session_idx] = results_rwck[4]
        beta_ck_array_rwck[session_idx] = results_rwck[5]
        nll_array_rwck[session_idx] = results_rwck[6]
        aic_array_rwck[session_idx] = results_rwck[7]
        bic_array_rwck[session_idx] = results_rwck[8]
        pseudo_r2_array_rwck[session_idx] = results_rwck[9]

        # Rescorla-Wagner Performance Delta (RWPD) model
        # Set bounds
        alpha_bound = (0, 1)  # Alpha
        sigma_bound = (1, 100)  # Standard deviation
        bias_bound = (0, 10)  # Mean at first trial
        w_rw_bound = (0, 10)  # Weight for RW update
        w_delta_p_bound = (0, 100)  # Weight for delta p
        bounds = [(alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1]),
                  (w_rw_bound[0], w_rw_bound[1]),
                  (w_delta_p_bound[0], w_delta_p_bound[1])]

        # Get results
        results_delta_p_rw = fit_model_with_cv(model=delta_P_RW,
                                       args=(confidence,
                                              feedback,
                                              n_trials,
                                              performance),
                                       bounds=bounds,
                                       n_trials=n_trials,
                                       start_value_number=50,
                                       solver="L-BFGS-B")

        alpha_array_delta_p_rw[session_idx] = results_delta_p_rw[0]
        sigma_array_delta_p_rw[session_idx] = results_delta_p_rw[1]
        bias_array_delta_p_rw[session_idx] = results_delta_p_rw[2]
        w_rw_array_delta_p_rw[session_idx] = results_delta_p_rw[3]
        w_delta_p_array_delta_p_rw[session_idx] = results_delta_p_rw[4]
        nll_array_delta_p_rw[session_idx] = results_delta_p_rw[5]
        aic_array_delta_p_rw[session_idx] = results_delta_p_rw[6]
        bic_array_delta_p_rw[session_idx] = results_delta_p_rw[7]
        pseudo_r2_array_delta_p_rw[session_idx] = results_delta_p_rw[8]



        # Define function to create a DataFrame for each model's session metrics
        def create_session_df(participant_idx,
                              session_idx,
                              session_metrics,
                              model_name):
            df = pd.DataFrame(session_metrics)
            df['Participant_ID'] = participant_idx
            df['Session_Number'] = session_idx
            df['Model_Name'] = model_name
            return df

# =============================================================================
#         # Define the metrics for each model
#         session_metrics_random = {'nll_array_random': nll_array_random[session_idx],
#                                   'aic_array_random': aic_array_random[session_idx],
#                                   'bic_array_random': bic_array_random[session_idx],
#                                   'pseudo_r2_array_random': pseudo_r2_array_random[session_idx]}
#         session_metrics_bias = {'mean_array_bias': mean_array_bias,#[session_idx],
#                                 'sd_array_bias': sd_array_bias,#[session_idx],
#                                 'nll_array_bias': nll_array_bias[session_idx],
#                                 'aic_array_bias': aic_array_bias[session_idx],
#                                 'bic_array_bias': bic_array_bias[session_idx],
#                                 'pseudo_r2_array_bias': pseudo_r2_array_bias[session_idx]}
#         session_metrics_wsls = {'std_WSLS_array': std_WSLS_array[session_idx],
#                                 'win_boundary_WSLS_array': win_boundary_WSLS_array[session_idx],
#                                 'nll_array_win_stay': nll_array_win_stay[session_idx],
#                                 'aic_array_win_stay': aic_array_win_stay[session_idx],
#                                 'bic_array_win_stay': bic_array_win_stay[session_idx],
#                                 'pseudo_r2_array_win_stay': pseudo_r2_array_win_stay[session_idx]}
#         session_metrics_rw_symm = {'alpha_array_rw_symm': alpha_array_rw_symm[session_idx],
#                                    'sigma_array_rw_symm': sigma_array_rw_symm[session_idx],
#                                    'bias_array_rw_symm': bias_array_rw_symm[session_idx],
#                                    'nll_array_rw_symm': nll_array_rw_symm[session_idx],
#                                    'aic_array_rw_symm': aic_array_rw_symm[session_idx],
#                                    'bic_array_rw_symm': bic_array_rw_symm[session_idx],
#                                    'pseudo_r2_array_rw_symm': pseudo_r2_array_rw_symm[session_idx]}
#         session_metrics_ck = {'alpha_array_ck': alpha_array_ck[session_idx],
#                               'sigma_array_ck': sigma_array_ck[session_idx],
#                               'bias_array_ck': bias_array_ck[session_idx],
#                               'beta_array_ck': beta_array_ck[session_idx],
#                               'nll_array_ck': nll_array_ck[session_idx],
#                               'aic_array_ck': aic_array_ck[session_idx],
#                               'bic_array_ck': bic_array_ck[session_idx],
#                               'pseudo_r2_array_ck': pseudo_r2_array_ck[session_idx]}
#         session_metrics_rwck = {'alpha_array_rwck': alpha_array_rwck[session_idx],
#                                 'alpha_ck_array_rwck': alpha_ck_array_rwck[session_idx],
#                                 'sigma_array_rwck': sigma_array_rwck[session_idx],
#                                 'bias_array_rwck': bias_array_rwck[session_idx],
#                                 'beta_array_rwck': beta_array_rwck[session_idx],
#                                 'beta_ck_array_rwck': beta_ck_array_rwck[session_idx],
#                                 'nll_array_rwck': nll_array_rwck[session_idx],
#                                 'aic_array_rwck': aic_array_rwck[session_idx],
#                                 'bic_array_rwck': bic_array_rwck[session_idx],
#                                 'pseudo_r2_array_rwck': pseudo_r2_array_rwck[session_idx]}
#         session_metrics_delta_p_rw = {'alpha_array_delta_p_rw': alpha_array_delta_p_rw[session_idx],
#                                       'sigma_array_delta_p_rw': sigma_array_delta_p_rw[session_idx],
#                                       'bias_array_delta_p_rw': bias_array_delta_p_rw[session_idx],
#                                       'w_rw_array_delta_p_rw': w_rw_array_delta_p_rw[session_idx],
#                                       'w_delta_p_array_delta_p_rw': w_delta_p_array_delta_p_rw[session_idx],
#                                       'nll_array_delta_p_rw': nll_array_delta_p_rw[session_idx],
#                                       'aic_array_delta_p_rw': aic_array_delta_p_rw[session_idx],
#                                       'bic_array_delta_p_rw': bic_array_delta_p_rw[session_idx],
#                                       'pseudo_r2_array_delta_p_rw': pseudo_r2_array_delta_p_rw[session_idx]}
#
# =============================================================================

        # Function to combine metrics into a single dictionary
        def create_combined_dict(participant_idx, session_idx, **metrics):
            combined_dict = {}
            for metric_name, metric_value in metrics.items():
                combined_dict.update(metric_value)
            combined_dict['Participant_ID'] = participant_idx
            combined_dict['Session_Number'] = session_idx
            return combined_dict

        # Ensure all metrics are lists or arrays
        session_metrics_random = {'nll_array_random': [nll_array_random[session_idx]],
                                  'aic_array_random': [aic_array_random[session_idx]],
                                  'bic_array_random': [bic_array_random[session_idx]],
                                  'pseudo_r2_array_random': [pseudo_r2_array_random[session_idx]]}
        session_metrics_bias = {'mean_array_bias': [mean_array_bias[session_idx]],
                                'sd_array_bias': [sd_array_bias[session_idx]],
                                'nll_array_bias': [nll_array_bias[session_idx]],
                                'aic_array_bias': [aic_array_bias[session_idx]],
                                'bic_array_bias': [bic_array_bias[session_idx]],
                                'pseudo_r2_array_bias': [pseudo_r2_array_bias[session_idx]]}
        session_metrics_wsls = {'std_WSLS_array': [std_WSLS_array[session_idx]],
                                'win_boundary_WSLS_array': [win_boundary_WSLS_array[session_idx]],
                                'nll_array_win_stay': [nll_array_win_stay[session_idx]],
                                'aic_array_win_stay': [aic_array_win_stay[session_idx]],
                                'bic_array_win_stay': [bic_array_win_stay[session_idx]],
                                'pseudo_r2_array_win_stay': [pseudo_r2_array_win_stay[session_idx]]}
        session_metrics_rw_symm = {'alpha_array_rw_symm': [alpha_array_rw_symm[session_idx]],
                                   'sigma_array_rw_symm': [sigma_array_rw_symm[session_idx]],
                                   'bias_array_rw_symm': [bias_array_rw_symm[session_idx]],
                                   'nll_array_rw_symm': [nll_array_rw_symm[session_idx]],
                                   'aic_array_rw_symm': [aic_array_rw_symm[session_idx]],
                                   'bic_array_rw_symm': [bic_array_rw_symm[session_idx]],
                                   'pseudo_r2_array_rw_symm': [pseudo_r2_array_rw_symm[session_idx]]}
        session_metrics_ck = {'alpha_array_ck': [alpha_array_ck[session_idx]],
                              'sigma_array_ck': [sigma_array_ck[session_idx]],
                              'bias_array_ck': [bias_array_ck[session_idx]],
                              'beta_array_ck': [beta_array_ck[session_idx]],
                              'nll_array_ck': [nll_array_ck[session_idx]],
                              'aic_array_ck': [aic_array_ck[session_idx]],
                              'bic_array_ck': [bic_array_ck[session_idx]],
                              'pseudo_r2_array_ck': [pseudo_r2_array_ck[session_idx]]}
        session_metrics_rwck = {'alpha_array_rwck': [alpha_array_rwck[session_idx]],
                                'alpha_ck_array_rwck': [alpha_ck_array_rwck[session_idx]],
                                'sigma_array_rwck': [sigma_array_rwck[session_idx]],
                                'bias_array_rwck': [bias_array_rwck[session_idx]],
                                'beta_array_rwck': [beta_array_rwck[session_idx]],
                                'beta_ck_array_rwck': [beta_ck_array_rwck[session_idx]],
                                'nll_array_rwck': [nll_array_rwck[session_idx]],
                                'aic_array_rwck': [aic_array_rwck[session_idx]],
                                'bic_array_rwck': [bic_array_rwck[session_idx]],
                                'pseudo_r2_array_rwck': [pseudo_r2_array_rwck[session_idx]]}
        session_metrics_delta_p_rw = {'alpha_array_delta_p_rw': [alpha_array_delta_p_rw[session_idx]],
                                      'sigma_array_delta_p_rw': [sigma_array_delta_p_rw[session_idx]],
                                      'bias_array_delta_p_rw': [bias_array_delta_p_rw[session_idx]],
                                      'w_rw_array_delta_p_rw': [w_rw_array_delta_p_rw[session_idx]],
                                      'w_delta_p_array_delta_p_rw': [w_delta_p_array_delta_p_rw[session_idx]],
                                      'nll_array_delta_p_rw': [nll_array_delta_p_rw[session_idx]],
                                      'aic_array_delta_p_rw': [aic_array_delta_p_rw[session_idx]],
                                      'bic_array_delta_p_rw': [bic_array_delta_p_rw[session_idx]],
                                      'pseudo_r2_array_delta_p_rw': [pseudo_r2_array_delta_p_rw[session_idx]]}

        # Combine all metrics into one dictionary
        combined_metrics = create_combined_dict(participant_idx, session_idx,
                                                session_metrics_random=session_metrics_random,
                                                session_metrics_bias=session_metrics_bias,
                                                session_metrics_wsls=session_metrics_wsls,
                                                session_metrics_rw_symm=session_metrics_rw_symm,
                                                session_metrics_ck=session_metrics_ck,
                                                session_metrics_rwck=session_metrics_rwck,
                                                session_metrics_delta_p_rw=session_metrics_delta_p_rw)

        # Create DataFrame for the combined metrics and append to results
        results.append(pd.DataFrame(combined_metrics))

# =============================================================================
#         # Define the save path
#         local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
#         working_dir = r'metacognition-learning\comparative_models'
#         save_path = r'results\variable_feedback\model_comparison'
#         name = 'all_sessions_model_metrics_CV.xlsx'
#         save_path_full = os.path.join(local_folder, working_dir, save_path,
#                                       name)
#
#         # Save the combined DataFrame to an Excel file
#         df_all_sessions.to_excel(save_path_full, index=False)
# =============================================================================

        # Increment session index
        session_idx += 1

    # Compute and store participant-level average metrics for Random Model
    nll_array_random_p[participant_idx] = np.mean(nll_array_random)
    aic_array_random_p[participant_idx] = np.mean(aic_array_random)
    bic_array_random_p[participant_idx] = np.mean(bic_array_random)
    pseudo_r2_array_random_p[participant_idx] = np.mean(pseudo_r2_array_random)

    # Compute and store participant-level average metrics for Bias Model
    mean_array_bias_p[participant_idx] = np.mean(mean_array_bias)
    sd_array_bias_p[participant_idx] = np.mean(sd_array_bias)
    nll_array_bias_p[participant_idx] = np.mean(nll_array_bias)
    aic_array_bias_p[participant_idx] = np.mean(aic_array_bias)
    bic_array_bias_p[participant_idx] = np.mean(bic_array_bias)
    pseudo_r2_array_bias_p[participant_idx] = np.mean(pseudo_r2_array_bias)

    # Compute and store participant-level average metrics for Win-Stay-Lose-Shift Model
    std_WSLS_array_p[participant_idx] = np.mean(std_WSLS_array)
    win_boundary_WSLS_array_p[participant_idx] = np.mean(win_boundary_WSLS_array)
    nll_array_win_stay_p[participant_idx] = np.mean(nll_array_win_stay)
    aic_array_win_stay_p[participant_idx] = np.mean(aic_array_win_stay)
    bic_array_win_stay_p[participant_idx] = np.mean(bic_array_win_stay)
    pseudo_r2_array_win_stay_p[participant_idx] = np.mean(pseudo_r2_array_win_stay)

    # Compute and store participant-level average metrics for RW Symmetric Model
    alpha_array_rw_symm_p[participant_idx] = np.mean(alpha_array_rw_symm)
    sigma_array_rw_symm_p[participant_idx] = np.mean(sigma_array_rw_symm)
    bias_array_rw_symm_p[participant_idx] = np.mean(bias_array_rw_symm)
    nll_array_rw_symm_p[participant_idx] = np.mean(nll_array_rw_symm)
    aic_array_rw_symm_p[participant_idx] = np.mean(aic_array_rw_symm)
    bic_array_rw_symm_p[participant_idx] = np.mean(bic_array_rw_symm)
    pseudo_r2_array_rw_symm_p[participant_idx] = np.mean(pseudo_r2_array_rw_symm)

    # Compute and store participant-level average metrics for Choice Kernel
    alpha_array_ck_p[participant_idx] = np.mean(alpha_array_ck)
    sigma_array_ck_p[participant_idx] = np.mean(sigma_array_ck)
    bias_array_ck_p[participant_idx] = np.mean(bias_array_ck)
    beta_array_ck_p[participant_idx] = np.mean(beta_array_ck)
    nll_array_ck_p[participant_idx] = np.mean(nll_array_ck)
    aic_array_ck_p[participant_idx] = np.mean(aic_array_ck)
    bic_array_ck_p[participant_idx] = np.mean(bic_array_ck)
    pseudo_r2_array_ck_p[participant_idx] = np.mean(pseudo_r2_array_ck)

    # Compute and store participant-level average metrics for RW + Choice Kernel
    alpha_array_rwck_p[participant_idx] = np.mean(alpha_array_rwck)
    alpha_ck_array_rwck_p[participant_idx] = np.mean(alpha_ck_array_rwck)
    sigma_array_rwck_p[participant_idx] = np.mean(sigma_array_rwck)
    bias_array_rwck_p[participant_idx] = np.mean(bias_array_rwck)
    beta_array_rwck_p[participant_idx] = np.mean(beta_array_rwck)
    beta_ck_array_rwck_p[participant_idx] = np.mean(beta_ck_array_rwck)
    nll_array_rwck_p[participant_idx] = np.mean(nll_array_rwck)
    aic_array_rwck_p[participant_idx] = np.mean(aic_array_rwck)
    bic_array_rwck_p[participant_idx] = np.mean(bic_array_rwck)
    pseudo_r2_array_rwck_p[participant_idx] = np.mean(pseudo_r2_array_rwck)

    # Compute and store participant-level average metrics for Delta P RW Model
    alpha_array_delta_p_rw_p[participant_idx] = np.mean(alpha_array_delta_p_rw)
    sigma_array_delta_p_rw_p[participant_idx] = np.mean(sigma_array_delta_p_rw)
    bias_array_delta_p_rw_p[participant_idx] = np.mean(bias_array_delta_p_rw)
    w_rw_array_delta_p_rw_p[participant_idx] = np.mean(w_rw_array_delta_p_rw)
    w_delta_p_array_delta_p_rw_p[participant_idx] = np.mean(w_delta_p_array_delta_p_rw)
    nll_array_delta_p_rw_p[participant_idx] = np.mean(nll_array_delta_p_rw)
    aic_array_delta_p_rw_p[participant_idx] = np.mean(aic_array_delta_p_rw)
    bic_array_delta_p_rw_p[participant_idx] = np.mean(bic_array_delta_p_rw)
    pseudo_r2_array_delta_p_rw_p[participant_idx] = np.mean(pseudo_r2_array_delta_p_rw)

    # Increment participant index
    participant_idx += 1

    # Save model metrics to excel file
    local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
    working_dir = r'metacognition-learning\comparative_models'
    save_path = r'results\variable_feedback\model_comparison'
    name = 'average_model_metrics_CV.xlsx'
    save_path_full = os.path.join(local_folder, working_dir, save_path, name)

    model_metric_dict = {

    # Participant
    'participant_idx': participant_idx,

    # Number of sessions completed
    'num_sessions': max_sessions,

    # Metrics for Random Model
    'nll_array_random_p': nll_array_random_p,
    'aic_array_random_p': aic_array_random_p,
    'bic_array_random_p': bic_array_random_p,
    'pseudo_r2_array_random_p': pseudo_r2_array_random_p,

    # Metrics for Bias Model
    'mean_array_bias_p': mean_array_bias_p,
    'sd_array_bias_p': sd_array_bias_p,
    'nll_array_bias_p': nll_array_bias_p,
    'aic_array_bias_p': aic_array_bias_p,
    'bic_array_bias_p': bic_array_bias_p,
    'pseudo_r2_array_bias_p': pseudo_r2_array_bias_p,

    # Metrics for Win-Stay-Lose-Shift Model
    'std_WSLS_array_p': std_WSLS_array_p,
    'win_boundary_WSLS_array_p': win_boundary_WSLS_array_p,
    'nll_array_win_stay_p': nll_array_win_stay_p,
    'aic_array_win_stay_p': aic_array_win_stay_p,
    'bic_array_win_stay_p': bic_array_win_stay_p,
    'pseudo_r2_array_win_stay_p': pseudo_r2_array_win_stay_p,

    # Metrics for RW Symmetric Model
    'alpha_array_rw_symm_p': alpha_array_rw_symm_p,
    'sigma_array_rw_symm_p': sigma_array_rw_symm_p,
    'bias_array_rw_symm_p': bias_array_rw_symm_p,
    'nll_array_rw_symm_p': nll_array_rw_symm_p,
    'aic_array_rw_symm_p': aic_array_rw_symm_p,
    'bic_array_rw_symm_p': bic_array_rw_symm_p,
    'pseudo_r2_array_rw_symm_p': pseudo_r2_array_rw_symm_p,

    # Metrics for choice kernel
    'alpha_array_ck_p': alpha_array_ck_p,
    'sigma_array_ck_p': sigma_array_ck_p,
    'bias_array_ck_p': bias_array_ck_p,
    'beta_array_ck_p': beta_array_ck_p,
    'nll_array_ck_p': nll_array_ck_p,
    'aic_array_ck_p': aic_array_ck_p,
    'bic_array_ck_p': bic_array_ck_p,
    'pseudo_r2_array_ck_p': pseudo_r2_array_ck_p,

    # Metrics for RW + choice kernel
    'alpha_array_rwck_p': alpha_array_rwck_p,
    'sigma_array_rwck_p': sigma_array_rwck_p,
    'bias_array_rwck_p': bias_array_rwck_p,
    'beta_array_rwck_p': beta_array_rwck_p,
    'beta_ck_array_rwck_p': beta_ck_array_rwck_p,
    'nll_array_rwck_p': nll_array_rwck_p,
    'aic_array_rwck_p': aic_array_rwck_p,
    'bic_array_rwck_p': bic_array_rwck_p,
    'pseudo_r2_array_rwck_p': pseudo_r2_array_rwck_p,

    # Metrics for Delta P RW Model
    'alpha_array_delta_p_rw_p': alpha_array_delta_p_rw_p,
    'sigma_array_delta_p_rw_p': sigma_array_delta_p_rw_p,
    'nll_array_delta_p_rw_p': nll_array_delta_p_rw_p,
    'aic_array_delta_p_rw_p': aic_array_delta_p_rw_p,
    'bic_array_delta_p_rw_p': bic_array_delta_p_rw_p,
    'pseudo_r2_array_delta_p_rw_p': pseudo_r2_array_delta_p_rw_p,
    }

    df_m = pd.DataFrame(model_metric_dict)
    df_m.to_excel(save_path_full)

    break

#%% Read the metrics back from the Excel file and assign to variables
local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
working_dir = r'metacognition-learning\comparative_models'
save_path = r'results\fixed_feedback\model_comparison'
name = 'model_metrics_CV_concatenated.xlsx'
save_path_full = os.path.join(local_folder, working_dir, save_path, name)
df_m = pd.read_excel(save_path_full)

# %% Get mean and sem

# Data for random model
random_model_mean_nll = np.mean(nll_array_random_p)
random_model_sem_nll = (np.std(nll_array_random_p) /
                        np.sqrt(len(nll_array_random_p)))
random_model_mean_aic = np.mean(aic_array_random_p)
random_model_sem_aic = np.std(aic_array_random_p) / \
                       np.sqrt(len(aic_array_random_p))
random_model_mean_bic = np.mean(bic_array_random_p)
random_model_sem_bic = np.std(bic_array_random_p) / \
                       np.sqrt(len(bic_array_random_p))

# Data for bias model
bias_model_mean_nll = np.mean(nll_array_bias_p)
bias_model_sem_nll = np.std(nll_array_bias_p) / \
                     np.sqrt(len(nll_array_bias_p))
bias_model_mean_aic = np.mean(aic_array_bias_p)
bias_model_sem_aic = np.std(aic_array_bias_p) / \
                     np.sqrt(len(aic_array_bias_p))
bias_model_mean_bic = np.mean(bic_array_bias_p)
bias_model_sem_bic = np.std(bic_array_bias_p) / \
                     np.sqrt(len(bic_array_bias_p))

# Data for win-stay model
win_stay_model_mean_nll = np.mean(nll_array_win_stay_p)
win_stay_model_sem_nll = np.std(nll_array_win_stay_p) / \
                         np.sqrt(len(nll_array_win_stay_p))
win_stay_model_mean_aic = np.mean(aic_array_win_stay_p)
win_stay_model_sem_aic = np.std(aic_array_win_stay_p) / \
                         np.sqrt(len(aic_array_win_stay_p))
win_stay_model_mean_bic = np.mean(bic_array_win_stay_p)
win_stay_model_sem_bic = np.std(bic_array_win_stay_p) / \
                         np.sqrt(len(bic_array_win_stay_p))

# Data for RW symmetric LR model
rw_symm_model_mean_nll = np.mean(nll_array_rw_symm_p)
rw_symm_model_sem_nll = np.std(nll_array_rw_symm_p) / \
                        np.sqrt(len(nll_array_rw_symm_p))
rw_symm_model_mean_aic = np.mean(aic_array_rw_symm_p)
rw_symm_model_sem_aic = np.std(aic_array_rw_symm_p) / \
                        np.sqrt(len(aic_array_rw_symm_p))
rw_symm_model_mean_bic = np.mean(bic_array_rw_symm_p)
rw_symm_model_sem_bic = np.std(bic_array_rw_symm_p) / \
                        np.sqrt(len(bic_array_rw_symm_p))

# Data for Choice Kernel (CK) model
ck_model_mean_nll = np.mean(nll_array_ck_p)
ck_model_sem_nll = np.std(nll_array_ck_p) / \
                   np.sqrt(len(nll_array_ck_p))
ck_model_mean_aic = np.mean(aic_array_ck_p)
ck_model_sem_aic = np.std(aic_array_ck_p) / \
                   np.sqrt(len(aic_array_ck_p))
ck_model_mean_bic = np.mean(bic_array_ck_p)
ck_model_sem_bic = np.std(bic_array_ck_p) / \
                   np.sqrt(len(bic_array_ck_p))

# Data for RW + Choice Kernel (RWCK) model
rwck_model_mean_nll = np.mean(nll_array_rwck_p)
rwck_model_sem_nll = np.std(nll_array_rwck_p) / \
                     np.sqrt(len(nll_array_rwck_p))
rwck_model_mean_aic = np.mean(aic_array_rwck_p)
rwck_model_sem_aic = np.std(aic_array_rwck_p) / \
                     np.sqrt(len(aic_array_rwck_p))
rwck_model_mean_bic = np.mean(bic_array_rwck_p)
rwck_model_sem_bic = np.std(bic_array_rwck_p) / \
                     np.sqrt(len(bic_array_rwck_p))



# Data for Delta P RW LR model
delta_p_rw_model_mean_nll = np.mean(nll_array_delta_p_rw_p)
delta_p_rw_model_sem_nll = np.std(nll_array_delta_p_rw_p) / \
                           np.sqrt(len(nll_array_delta_p_rw_p))
delta_p_rw_model_mean_aic = np.mean(aic_array_delta_p_rw_p)
delta_p_rw_model_sem_aic = np.std(aic_array_delta_p_rw_p) / \
                           np.sqrt(len(aic_array_delta_p_rw_p))
delta_p_rw_model_mean_bic = np.mean(bic_array_delta_p_rw_p)
delta_p_rw_model_sem_bic = np.std(bic_array_delta_p_rw_p) / \
                           np.sqrt(len(bic_array_delta_p_rw_p))

#%%# Function to calculate mean and SEM
#%   Metrics: Absolute model fit

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# Dictionary to store the results
results = {}

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_array_random_p', 'aic_array_random_p', 'bic_array_random_p'],
    ['bias', 'nll_array_bias_p', 'aic_array_bias_p', 'bic_array_bias_p'],
    ['win_stay', 'nll_array_win_stay_p', 'aic_array_win_stay_p', 'bic_array_win_stay_p'],
    #['rw_static', 'nll_array_rw_static_p', 'aic_array_rw_static_p', 'bic_array_rw_static_p'],
    ['rw_symm', 'nll_array_rw_symm_p', 'aic_array_rw_symm_p', 'bic_array_rw_symm_p'],
    #['rw_cond', 'nll_array_rw_cond_p', 'aic_array_rw_cond_p', 'bic_array_rw_cond_p'],
    ['ck', 'nll_array_ck_p', 'aic_array_ck_p', 'bic_array_ck_p'],
    ['rwck', 'nll_array_rwck_p', 'aic_array_rwck_p', 'bic_array_rwck_p'],
    ['delta_p_rw', 'nll_array_delta_p_rw_p', 'aic_array_delta_p_rw_p', 'bic_array_delta_p_rw_p'],
]

# Loop through each model and metric, calculate mean and SEM, and store in results dictionary
for model, nll_col, aic_col, bic_col in models_metrics:
    results[f'{model}_model_mean_nll'], results[f'{model}_model_sem_nll'] = calculate_mean_sem(df_m[nll_col])
    results[f'{model}_model_mean_aic'], results[f'{model}_model_sem_aic'] = calculate_mean_sem(df_m[aic_col])
    results[f'{model}_model_mean_bic'], results[f'{model}_model_sem_bic'] = calculate_mean_sem(df_m[bic_col])

# Loop through each model and metric, calculate mean and SEM, and create named variables
for model, nll_col, aic_col, bic_col in models_metrics:
    # Create variables for each array
    exec(f'{nll_col} = df_m["{nll_col}"].values')
    exec(f'{aic_col} = df_m["{aic_col}"].values')
    exec(f'{bic_col} = df_m["{bic_col}"].values')

    # Calculate mean and SEM
    mean_nll, sem_nll = calculate_mean_sem(eval(nll_col))
    mean_aic, sem_aic = calculate_mean_sem(eval(aic_col))
    mean_bic, sem_bic = calculate_mean_sem(eval(bic_col))

    # Create variables for mean and SEM
    exec(f'{model}_model_mean_nll = {mean_nll}')
    exec(f'{model}_model_sem_nll = {sem_nll}')
    exec(f'{model}_model_mean_aic = {mean_aic}')
    exec(f'{model}_model_sem_aic = {sem_aic}')
    exec(f'{model}_model_mean_bic = {mean_bic}')
    exec(f'{model}_model_sem_bic = {sem_bic}')

#%% Relative fit

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_array_random_p', 'aic_array_random_p', 'bic_array_random_p'],
    ['bias', 'nll_array_bias_p', 'aic_array_bias_p', 'bic_array_bias_p'],
    ['win_stay', 'nll_array_win_stay_p', 'aic_array_win_stay_p', 'bic_array_win_stay_p'],
    #['rw_static', 'nll_array_rw_static_p', 'aic_array_rw_static_p', 'bic_array_rw_static_p'],
    ['rw_symm', 'nll_array_rw_symm_p', 'aic_array_rw_symm_p', 'bic_array_rw_symm_p'],
    #['rw_cond', 'nll_array_rw_cond_p', 'aic_array_rw_cond_p', 'bic_array_rw_cond_p'],
    ['ck', 'nll_array_ck_p', 'aic_array_ck_p', 'bic_array_ck_p'],
    ['rwck', 'nll_array_rwck_p', 'aic_array_rwck_p', 'bic_array_rwck_p'],
    ['delta_p_rw', 'nll_array_delta_p_rw_p', 'aic_array_delta_p_rw_p', 'bic_array_delta_p_rw_p'],
]

# Dictionary to store the results
results = {}

# Adjust the fits for each participant based on the best model for each metric
df_m['participant_id'] = df_m.index.values
participants = df_m['participant_id'].unique()
for participant in participants:
    for metric_index in range(1, 4):  # Index 1 for NLL, 2 for AIC, 3 for BIC
        min_metric_value = float('inf')
        best_model_metric_col = None

        # Find the best model based on the current metric for this participant
        for _, *metrics in models_metrics:
            participant_metric = df_m.loc[df_m['participant_id'] == participant, metrics[metric_index-1]].values
            if participant_metric.min() < min_metric_value:
                min_metric_value = participant_metric.min()
                best_model_metric_col = metrics[metric_index-1]

        # Subtract the best model's metric from all models' metrics for this participant and the current metric
        for _, *metrics in models_metrics:
            df_m.loc[df_m['participant_id'] == participant, metrics[metric_index-1]] -= min_metric_value

# Calculate mean and SEM for the adjusted metrics
for model, nll_col, aic_col, bic_col in models_metrics:
    # For NLL
    results[f'{model}_model_mean_nll'], results[f'{model}_model_sem_nll'] = calculate_mean_sem(df_m[nll_col])
    # For AIC
    results[f'{model}_model_mean_aic'], results[f'{model}_model_sem_aic'] = calculate_mean_sem(df_m[aic_col])
    # For BIC
    results[f'{model}_model_mean_bic'], results[f'{model}_model_sem_bic'] = calculate_mean_sem(df_m[bic_col])

    # Create variables for each array
    exec(f'{nll_col} = df_m["{nll_col}"].values')
    exec(f'{aic_col} = df_m["{aic_col}"].values')
    exec(f'{bic_col} = df_m["{bic_col}"].values')

    # Calculate mean and SEM
    mean_nll, sem_nll = calculate_mean_sem(eval(nll_col))
    mean_aic, sem_aic = calculate_mean_sem(eval(aic_col))
    mean_bic, sem_bic = calculate_mean_sem(eval(bic_col))

    # Create variables for mean and SEM
    exec(f'{model}_model_mean_nll = {mean_nll}')
    exec(f'{model}_model_sem_nll = {sem_nll}')
    exec(f'{model}_model_mean_aic = {mean_aic}')
    exec(f'{model}_model_sem_aic = {sem_aic}')
    exec(f'{model}_model_mean_bic = {mean_bic}')
    exec(f'{model}_model_sem_bic = {sem_bic}')


#%%  Histogram of best model

metric = [
    [nll_array_random_p,
    nll_array_bias_p,
    nll_array_win_stay_p,
    nll_array_rw_symm_p,
    nll_array_ck_p,
    nll_array_rwck_p,
    nll_array_delta_p_rw_p,],

    [aic_array_random_p,
    aic_array_bias_p,
    aic_array_win_stay_p,
    aic_array_rw_symm_p,
    aic_array_ck_p,
    aic_array_rwck_p,
    aic_array_delta_p_rw_p],

    [bic_array_random_p,
    bic_array_bias_p,
    bic_array_win_stay_p,
    bic_array_rw_symm_p,
    bic_array_ck_p,
    bic_array_rwck_p,
    bic_array_delta_p_rw_p]

          ]

# Loop over metrics
fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize=(6,6))
for metric_list, metric_name, ax in zip(metric,
                                        ['NLL', 'AIC', 'BIC'],
                                        [ax, ax2, ax3]):

    score_board = []
    pids = []
    # Loop over each participant's model score
    for rand, bias, wsls, rw, ck, rwck, delta_p_rw, pid in zip(metric_list[0],
                                                               metric_list[1],
                                                               metric_list[2],
                                                               metric_list[3],
                                                               metric_list[4],
                                                               metric_list[5],
                                                               metric_list[6],
                                                     range(len(metric_list[6]))
                                                               ):

        # Scores from different models
        scores = np.array([rand, bias, wsls, rw, ck, rwck, delta_p_rw])

        # Find the minimum score
        min_score = np.min(scores)

        # Get indices of all occurrences of the lowest score
        # e.g., if multiple models have the lowest score, take both
        idxs = np.where(scores == min_score)[0]

        # Save best models - all models with the lowest score
        for idx in idxs:
            score_board.append(idx)
            pids.append(pid)

    # Get pid of participants with RWPD as best model according to NLL
    if metric_name == 'NLL':
        df_best_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
        pid_wsls_best = df_best_fit[df_best_fit.best_model==2].pid
        pid_rw_best = df_best_fit[df_best_fit.best_model==3].pid
        pid_rwpd_best = df_best_fit[df_best_fit.best_model==6].pid
    else:
        pass

    models = ['random', 'biased', 'WSLS', 'RW',
              'CK', 'RWCK', 'RWPD']
    counts = [score_board.count(0), # Rand
              score_board.count(1), # Bias
              score_board.count(2), # WSLS
              score_board.count(3), # RW
              score_board.count(4), # CK
              score_board.count(5), # RWCK
              score_board.count(6),] #RWPD

    bar_colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
    bars = ax.bar(models, counts, color=bar_colors)

    # Customizing the Axes
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_ylabel('n participants')
    ax.set_xlim(-1, len(models))
    ax.set_title(metric_name)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.suptitle('Best Model')
plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_hist_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)
plt.show()


# %% Plot mean+-sem across conditions
from scipy.stats import ttest_rel


def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    """
    y_max = max(max(y1_values), max(y2_values)) * 4.1
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')

    # Draw vertical lines
    ax.plot([x1, x1], [y_max-(0.2*y_max), y_max ], color='black')
    ax.plot([x2, x2], [y_max-(0.2*y_max), y_max ], color='black')

    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')


fig, ax = plt.subplots(figsize=(8, 4))  # Increase the figure size for better readability

# Define x-coordinates and offsets for each model within the group
x = np.arange(0, 9, 3)  # Base x-coordinates for metrics
offset = 0.35  # Offset for each model within a group
capsize = 8

# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # Different markers for each model

# Plotting NLL for each model
model_values = [
    nll_array_random_p, nll_array_bias_p,
    nll_array_win_stay_p, nll_array_rw_symm_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p,
]
model_means = [
    random_model_mean_nll, bias_model_mean_nll,
    win_stay_model_mean_nll, rw_symm_model_mean_nll,
    ck_model_mean_nll, rwck_model_mean_nll,
    delta_p_rw_model_mean_nll,
]
model_sems = [
    random_model_sem_nll, bias_model_sem_nll,
    win_stay_model_sem_nll, rw_symm_model_sem_nll,
    ck_model_sem_nll, rwck_model_sem_nll,
    delta_p_rw_model_sem_nll,
]
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW Symmetric LR", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

for i, (values, mean, sem) in enumerate(zip(model_values, model_means, model_sems)):
    # Error bar
    ax.errorbar(
        x[0] + offset * (i - 1.5), mean, yerr=sem,
        fmt=markers[i], capsize=capsize, color=colors[i], label=model_names[i],
         markeredgecolor='black', markeredgewidth=1  # Add black outline
    )
    # Scatter plot of individual samples
    ax.scatter(
        np.full(values.shape, x[0] + offset * (i - 1.5)),
        values, color=colors[i], alpha=0.7, s=2, marker=markers[i], label='_Hidden label'
    )

# Perform paired samples t-tests for NLL data
_, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
_, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)

# Annotate significance
annotate_significance(ax, x[0] + offset * (3 - 1.5), x[0] + offset * (2 - 1.5), nll_array_rw_symm_p, nll_array_win_stay_p, p_value_winstay)
annotate_significance(ax, x[0] + offset * (3 - 1.5), x[0] + offset * (6 - 1.5), nll_array_rw_symm_p, nll_array_delta_p_rw_p, p_value_rwpd)

# Plotting AIC for each model
model_values = [
    aic_array_random_p, aic_array_bias_p,
    aic_array_win_stay_p, aic_array_rw_symm_p,
    aic_array_ck_p, aic_array_rwck_p,
    aic_array_delta_p_rw_p,
]
model_means = [
    random_model_mean_aic, bias_model_mean_aic,
    win_stay_model_mean_aic, rw_symm_model_mean_aic,
    ck_model_mean_aic, rwck_model_mean_aic,
    delta_p_rw_model_mean_aic,
]
model_sems = [
    random_model_sem_aic, bias_model_sem_aic,
    win_stay_model_sem_aic, rw_symm_model_sem_aic,
    ck_model_sem_aic, rwck_model_sem_aic,
    delta_p_rw_model_sem_aic,
]

for i, (values, mean, sem) in enumerate(zip(model_values, model_means, model_sems)):
    # Error bar
    ax.errorbar(
        x[1] + offset * (i - 1.5), mean, yerr=sem,
        fmt=markers[i], capsize=capsize, color=colors[i],
        markeredgecolor='black', markeredgewidth=1
    )
    # Scatter plot of individual samples
    ax.scatter(
        np.full(values.shape, x[1] + offset * (i - 1.5)),
        values, color=colors[i], alpha=0.7, s=2, marker=markers[i], label='_Hidden label'
    )

# Plotting BIC for each model
model_values = [
    bic_array_random_p, bic_array_bias_p,
    bic_array_win_stay_p, bic_array_rw_symm_p,
    bic_array_ck_p, bic_array_rwck_p,
    bic_array_delta_p_rw_p,
]
model_means = [
    random_model_mean_bic, bias_model_mean_bic,
    win_stay_model_mean_bic, rw_symm_model_mean_bic,
    ck_model_mean_bic, rwck_model_mean_bic,
    delta_p_rw_model_mean_bic,
]
model_sems = [
    random_model_sem_bic, bias_model_sem_bic,
    win_stay_model_sem_bic, rw_symm_model_sem_bic,
    ck_model_sem_bic, rwck_model_sem_bic,
    delta_p_rw_model_sem_bic,
]

for i, (values, mean, sem) in enumerate(zip(model_values, model_means, model_sems)):
    # Error bar
    ax.errorbar(
        x[2] + offset * (i - 1.5), mean, yerr=sem,
        fmt=markers[i], capsize=capsize, color=colors[i],
        markeredgecolor='black', markeredgewidth=1
    )
    # Scatter plot of individual samples
    ax.scatter(
        np.full(values.shape, x[2] + offset * (i - 1.5)),
        values, color=colors[i], alpha=0.7, s=2, marker=markers[i], label='_Hidden label'
    )

# Customizing the Axes
ax.set_xticks(x)
ax.set_xticklabels(['NLL', 'AIC', 'BIC'])
ax.set_ylabel('Model fits')
ax.set_yscale('log')  # Set y-axis to log scale

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adding a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, scatterpoints=1, markerscale=1.2,
          fontsize='small', title='Models')

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()

#%% using whisker plots


def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value, heightOffsetScalar=2.2):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    """
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (0.05 * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (0.05 * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')


fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size for better readability

# Define x-coordinates and offsets for each model within the group
x = np.arange(0, 9, 3)  # Base x-coordinates for metrics
offset = 0.35  # Offset for each model within a group

# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # Different markers for each model

# Plotting NLL for each model
model_values = [
    nll_array_random_p, nll_array_bias_p,
    nll_array_win_stay_p, nll_array_rw_symm_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p,
]
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW Symmetric LR", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[0] + offset * (i - 1.5)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black')
    )
    # Scatter plot of individual samples
    ax.scatter(
        np.full(values.shape, x[0] + offset * (i - 1.5)),
        values, color=colors[i], alpha=0.7, s=20, marker=markers[i], label='_Hidden label'
    )

# Perform paired samples t-tests for NLL data
_, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
_, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)

# Annotate significance
annotate_significance(ax, x[0] + offset * (3 - 1.5), x[0] + offset * (2 - 1.5), nll_array_rw_symm_p, nll_array_win_stay_p, p_value_winstay, 2.2)
annotate_significance(ax, x[0] + offset * (3 - 1.5), x[0] + offset * (6 - 1.5), nll_array_rw_symm_p, nll_array_delta_p_rw_p, p_value_rwpd, 2.2)

# Plotting AIC for each model
model_values = [
    aic_array_random_p, aic_array_bias_p,
    aic_array_win_stay_p, aic_array_rw_symm_p,
    aic_array_ck_p, aic_array_rwck_p,
    aic_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[1] + offset * (i - 1.5)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black')
    )
    # Scatter plot of individual samples
    ax.scatter(
        np.full(values.shape, x[1] + offset * (i - 1.5)),
        values, color=colors[i], alpha=0.7, s=20, marker=markers[i], label='_Hidden label'
    )

# Plotting BIC for each model
model_values = [
    bic_array_random_p, bic_array_bias_p,
    bic_array_win_stay_p, bic_array_rw_symm_p,
    bic_array_ck_p, bic_array_rwck_p,
    bic_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[2] + offset * (i - 1.5)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black')
    )
    # Scatter plot of individual samples
    ax.scatter(
        np.full(values.shape, x[2] + offset * (i - 1.5)),
        values, color=colors[i], alpha=0.7, s=20, marker=markers[i], label='_Hidden label'
    )

# Customizing the Axes
ax.set_xticks(x)
ax.set_xticklabels(['NLL', 'AIC', 'BIC'])
ax.set_ylabel('Model fits')
#ax.set_yscale('log')  # Set y-axis to log scale
ax.set_title('Model Comparison Across Metrics')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)


# Adding a legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[:len(model_names)], model_names,
          scatterpoints=1, markerscale=1.2, fontsize='small', title='Models')

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()

#%% Only NLL
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, shapiro
import seaborn as sns
import matplotlib.colors as mcolors

def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value, heightOffsetScalar=2.2, lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    """
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

def darken_color(color, amount=0.5):
    """
    Darken a given color.
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([max(0, min(1, c[i] * (1 - amount))) for i in range(3)])

fig, ax = plt.subplots(figsize=(7, 5))  # Increase the figure size for better readability

model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]
# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # Different markers for each model

# Define x-coordinates and offsets for each model within the group
x = np.arange(0, 1)  # Base x-coordinates for metrics
offset = 0.35  # Offset for each model within a group

# Plotting NLL for each model
model_values = [
    nll_array_random_p, nll_array_bias_p,
    nll_array_win_stay_p, nll_array_rw_symm_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[0] + offset * (i - 3)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black'), zorder=0)

# =============================================================================
#     # Scatter plot of individual samples
#     darker_color = darken_color(colors[i], amount=0.3)  # Darken the color
#     ax.scatter(np.full(values.shape, x[0] + offset * (i - 3)),
#                values, color=darker_color, alpha=0.3, s=20, marker=markers[i],
#                label=f'{model_names[i]}', zorder=1,
#                #edgecolor='grey',
#                )
#
# =============================================================================
# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (-1),
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, 1.5, 0.1)
annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (3),
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, 2.2, 0.035)

# Customizing the Axes
ax.set_xticks([x[0]])
ax.set_xticklabels([''])
ax.set_xlabel('Cross-validated NLL')
ax.set_ylabel('Model fits')
#ax.set_yscale('log')  # Set y-axis to log scale

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

import matplotlib.lines as mlines

# Custom legend handles
custom_handles = [
    mlines.Line2D([], [], color=color, marker='o',
                  linestyle='None', markersize=6,
                  markerfacecolor=color,
                  alpha=1)
    for color in colors
]

# Adding a legend
ax.legend(custom_handles, model_names, scatterpoints=1, markerscale=1.2, fontsize='small',
          loc='upper left', bbox_to_anchor=(0.9, 1))
# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()


#%% Plot the alpha of RWPD vs BDI
import scipy.stats as stats

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.alpha_array_delta_p_rw_p[pid_rwpd_best.values]
    y = bdi[pid_rwpd_best.values]
else:
    x = df_m.alpha_array_delta_p_rw_p
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.045, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Add legend
#ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% Plot sigma of RWPD vs BDI
import scipy.stats as stats

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.sigma_array_delta_p_rw_p[pid_rwpd_best.values]
    y = bdi[pid_rwpd_best.values]
else:
    x = df_m.sigma_array_delta_p_rw_p
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('Standard Deviation in Model Estimates')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Add legend
#ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% Plot the alpha of RW vs BDI
import scipy.stats as stats

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.alpha_array_rw_symm_p[pid_rw_best.values]
    y = bdi[pid_rw_best.values]
else:
    x = df_m.alpha_array_rw_symm_p
    y = bdi
# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')
ax.set_xlim(-0.05, 0.55)
# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.08, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Add legend
#ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'RW_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()


#%% Plot the wsls win boundry vs BDI
import scipy.stats as stats

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:68], total=len(df.pid.unique()[:68])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)
x = df_m.win_boundary_WSLS_array_p[pid_wsls_best.values]
y = bdi[pid_wsls_best.values]

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('WSLS win boundary')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Add legend
ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'WSLS_win_boundary_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()
#%% GLS regression of alpha vs bdi
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

bdi = []
group_level_bias = []
confidence = []
group_level_confidence_mean = df[df.condition!='baseline'].confidence.mean()
for participant in tqdm(df['pid'].unique(), total=len(df['pid'].unique())):

    # filter on participant
    df_p = df[df.pid==participant]
    # Get bdi
    bdi.append(df_p['bdi'].unique()[0])

    # Get participant confidence
    participant_c = df_p[df_p.condition!='baseline'].confidence.mean()
    confidence.append(participant_c)
    bias = participant_c-group_level_confidence_mean
    group_level_bias.append(bias)

# Extracting the relevant data
bdi = np.array(bdi)
confidence = np.array(confidence)
x = alpha_array_delta_p_rw_p[pid_rwpd_best.values
                                ]
x2 = confidence[pid_rwpd_best.values
                ]
y = bdi[pid_rwpd_best.values
        ]

# Prepare the data for OLS regression
x_with_const = sm.add_constant(x)

# Initial OLS regression to estimate weights
model_ols = sm.OLS(y, x_with_const).fit()
print(model_ols.summary())

# Compute absolute residuals from the initial OLS model
absolute_residuals = np.abs(model_ols.resid)

# Regress absolute residuals on the independent variable to model the variance
model_resid = sm.OLS(absolute_residuals, x_with_const).fit()

# Use the inverse of the predicted values from this model as weights for GLS
weights = 1 / model_resid.predict(x_with_const)

# Fit the GLS model with the estimated weights
gls_model = sm.GLS(y, x_with_const, weights=weights)
gls_results = gls_model.fit()

# Extract coefficients and p-values from the GLS model results
coefficients = gls_results.params
p_values = gls_results.pvalues

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(x, y)

# Add the trendline from GLS results
ax.plot(x, coefficients[0] + coefficients[1] * x, color='red')

# Annotate with R^2, coefficients, and p-values
annotation_text = (f'Intercept={coefficients[0]:.2f} (p={p_values[0]:.3f})\n'
                   f'Slope={coefficients[1]:.2f} (p={p_values[1]:.3f})')

ax.annotate(annotation_text,
            xy=(0.60, 0.95), xycoords='axes fraction',
            fontsize=10, color='red', verticalalignment='top')

ax.set_xlabel('Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

# Print the summary of the GLS model
print(gls_results.summary())

#%% now plot metacog bias  (mean confidence)

# Create a set of all indices from 0 to 131
all_indices = set(np.arange(132))

# Create a set from pid_rwpd_best.values
pid_indices_set = set(pid_rwpd_best.values)

# Find the difference between the two sets to get indices that are not in pid_rwpd_best.values
missing_indices = np.array(list(all_indices - pid_indices_set))


# Extracting the relevant data
confidence = np.array(confidence)
group_level_bias = np.array(group_level_bias)
x = group_level_bias[pid_rwpd_best.values]
y = bdi[pid_rwpd_best.values]

# Prepare the data for OLS regression
x_with_const = sm.add_constant(x)

# Initial OLS regression to estimate weights
model_ols = sm.OLS(y, x_with_const).fit()
print(model_ols.summary())

# Compute absolute residuals from the initial OLS model
absolute_residuals = np.abs(model_ols.resid)

# Regress absolute residuals on the independent variable to model the variance
model_resid = sm.OLS(absolute_residuals, x_with_const).fit()

# Use the inverse of the predicted values from this model as weights for GLS
weights = 1 / model_resid.predict(x_with_const)

# Fit the GLS model with the estimated weights
gls_model = sm.GLS(y, x_with_const, weights=weights)
gls_results = gls_model.fit()

# Extract coefficients and p-values from the GLS model results
coefficients = gls_results.params
p_values = gls_results.pvalues


# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(x, y)

# Add the trendline from GLS results
ax.plot(x, coefficients[0] + coefficients[1] * x, color='red')

# Annotate with R^2, coefficients, and p-values
annotation_text = (f'Intercept={coefficients[0]:.2f} (p={p_values[0]:.3f})\n'
                   f'Slope={coefficients[1]:.2f} (p={p_values[1]:.3f})')

ax.annotate(annotation_text,
            xy=(0.60, 0.95), xycoords='axes fraction',
            fontsize=10, color='red', verticalalignment='top')

ax.set_xlabel('Metacognitive bias')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'Confidence_mean_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import statsmodels.api as sm

# Assuming other parts of the script (data loading, bdi, and confidence calculation) are unchanged...

# Extracting the relevant data
bdi = np.array(bdi)
confidence = np.array(confidence)
x1 = alpha_array_delta_p_rw_p[pid_rwpd_best.values]  # First regressor
x2 = confidence[pid_rwpd_best.values]  # Second regressor
y = bdi[pid_rwpd_best.values]  # Dependent variable

# Prepare the data for regression, combining x1 and x2 with an intercept
X = np.column_stack((x1, x2))
X_with_const = sm.add_constant(X)

# Initial OLS regression to estimate weights, using both regressors
model_ols = sm.OLS(y, X_with_const).fit()

# Compute absolute residuals from the initial OLS model
absolute_residuals = np.abs(model_ols.resid)

# Regress absolute residuals on the independent variables to model the variance
model_resid = sm.OLS(absolute_residuals, X_with_const).fit()

# Use the inverse of the predicted values from this model as weights for GLS
weights = 1 / model_resid.predict(X_with_const)

# Fit the GLS model with the estimated weights and both regressors
gls_model = sm.GLS(y, X_with_const, weights=weights)
gls_results = gls_model.fit()

# Create the scatter plot for the first regressor
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(x1, y)  # Note: This scatter plot only shows the relationship between x1 and y

# Add the trendline from GLS results for the first regressor
# Note: This is a simplification since the model includes another regressor
ax.plot(x1, gls_results.params[0] + gls_results.params[1] * x1, color='red')

# Annotate with coefficients and p-values from the GLS model
annotation_text = (f'Intercept={gls_results.params[0]:.2f} (p={gls_results.pvalues[0]:.3f})\n'
                   f'Learning Rate Coeff.={gls_results.params[1]:.2f} (p={gls_results.pvalues[1]:.3f})\n'
                   f'Confidence Coeff.={gls_results.params[2]:.2f} (p={gls_results.pvalues[2]:.3f})')

ax.annotate(annotation_text,
            xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=10, color='red', verticalalignment='top')

ax.set_xlabel('Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Save and show plot as before...

# Print the summary of the GLS model
print(gls_results.summary())

#%%

#%% Plot the alpha of RWPD vs BDI
import scipy.stats as stats

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

bdi = []
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):
    # Get bdi
    bdi.append(df[df.pid==participant].bdi_score.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)
best_idx = pid_rwpd_best
x = df_m.alpha_array_delta_p_rw_p[:]
y = bdi[:]

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
print(f'slope: {round(slope,4)}\nintercept: {round(intercept,4)}')
print(f'r_value: {round(r_value,4)}\np_value: {round(p_value,4)}')

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(x, y, label='Data points')

# Plot the regression line
regression_line = slope * x + intercept
ax.plot(x, regression_line, color='red', label='Regression line')

ax.set_xlabel('Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)
ax.legend()

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI_v2_fixed_feedback.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the figure
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

