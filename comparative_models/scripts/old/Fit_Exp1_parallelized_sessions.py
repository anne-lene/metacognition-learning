# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:28:24 2024

@author: carll
"""


# Fit models to Experiment 1 - Fixed Feedback

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys
from multiprocessing import Pool, cpu_count
import logging

# Set the current working directory one level up from the 'scripts' directory
current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(current_directory)

# Add the 'src' directory to the Python path
src_directory = os.path.join(current_directory, 'src')
sys.path.append(src_directory)

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

# Function to process data for a single session
def process_session(pid_session_tuple, df):

    # Participant and session
    participant, session = pid_session_tuple

    # Current session data only
    df_s = df[(df['pid'] == participant) | (df['session'] == session)]

    # n sessions
    max_sessions = 1
    session_idx = 0

    # Pre-allocate arrays for session-level metrics
    nll_array_random = np.zeros(max_sessions)
    aic_array_random = np.zeros(max_sessions)
    bic_array_random = np.zeros(max_sessions)
    pseudo_r2_array_random = np.zeros(max_sessions)

    # Other arrays for Bias Model, WSLS Model, etc., similar to the provided code
    mean_array_bias = np.zeros(max_sessions)
    sd_array_bias = np.zeros(max_sessions)
    nll_array_bias = np.zeros(max_sessions)
    aic_array_bias = np.zeros(max_sessions)
    bic_array_bias = np.zeros(max_sessions)
    pseudo_r2_array_bias = np.zeros(max_sessions)

    std_WSLS_array = np.zeros(max_sessions)
    win_boundary_WSLS_array = np.zeros(max_sessions)
    nll_array_win_stay = np.zeros(max_sessions)
    aic_array_win_stay = np.zeros(max_sessions)
    bic_array_win_stay = np.zeros(max_sessions)
    pseudo_r2_array_win_stay = np.zeros(max_sessions)

    alpha_array_rw_symm = np.zeros(max_sessions)
    sigma_array_rw_symm = np.zeros(max_sessions)
    bias_array_rw_symm = np.zeros(max_sessions)
    nll_array_rw_symm = np.zeros(max_sessions)
    aic_array_rw_symm = np.zeros(max_sessions)
    bic_array_rw_symm = np.zeros(max_sessions)
    pseudo_r2_array_rw_symm = np.zeros(max_sessions)

    alpha_array_ck = np.zeros(max_sessions)
    sigma_array_ck = np.zeros(max_sessions)
    bias_array_ck = np.zeros(max_sessions)
    beta_array_ck = np.zeros(max_sessions)
    nll_array_ck = np.zeros(max_sessions)
    aic_array_ck = np.zeros(max_sessions)
    bic_array_ck = np.zeros(max_sessions)
    pseudo_r2_array_ck = np.zeros(max_sessions)

    alpha_array_rwck = np.zeros(max_sessions)
    alpha_ck_array_rwck = np.zeros(max_sessions)
    sigma_array_rwck = np.zeros(max_sessions)
    bias_array_rwck = np.zeros(max_sessions)
    beta_array_rwck = np.zeros(max_sessions)
    ck_beta_array_rwck = np.zeros(max_sessions)
    nll_array_rwck = np.zeros(max_sessions)
    aic_array_rwck = np.zeros(max_sessions)
    bic_array_rwck = np.zeros(max_sessions)
    pseudo_r2_array_rwck = np.zeros(max_sessions)

    alpha_array_delta_p_rw = np.zeros(max_sessions)
    sigma_array_delta_p_rw = np.zeros(max_sessions)
    bias_array_delta_p_rw = np.zeros(max_sessions)
    w_rw_array_delta_p_rw = np.zeros(max_sessions)
    w_delta_p_array_delta_p_rw = np.zeros(max_sessions)
    nll_array_delta_p_rw = np.zeros(max_sessions)
    aic_array_delta_p_rw = np.zeros(max_sessions)
    bic_array_delta_p_rw = np.zeros(max_sessions)
    pseudo_r2_array_delta_p_rw = np.zeros(max_sessions)

    # Remove baseline
    df_s = df_s[df_s.condition != 'baseline']

    # Calculate trial error as average across subtrials
    df_s['difference'] = df_s['estimate'] - df_s['correct']
    error_avg = df_s.groupby('trial')['difference'].mean()

    # Only keep first row of every subtrial (one row = one trial)
    df_s = df_s.drop_duplicates(subset='trial', keep='first')

    # Condition
    condition = df_s.condition.unique()[0]

    # N trials
    n_trials = len(df_s)

    # Calculate trial-by-trial metrics
    confidence = df_s.confidence.values
    feedback = df_s.feedback.values
    performance = -error_avg.values

    # Random confidence model
    bounds = [(0, 0)]
    results = fit_model_with_cv(model=random_model,
                                args=(100, n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")
    x = results[0]
    nll_array_random[session_idx] = results[1]
    aic_array_random[session_idx] = results[2]
    bic_array_random[session_idx] = results[3]
    pseudo_r2_array_random[session_idx] = results[4]

    # Biased confidence model
    mean_bound = (0, 100)
    sigma_bound = (1, 100)
    bounds = [(mean_bound[0], mean_bound[1]),
              (sigma_bound[0], sigma_bound[1])]
    results = fit_model_with_cv(model=random_model_w_bias,
                                args=(confidence, n_trials),
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

    # Win-stay-lose-shift model
    sigma_bound = (1, 100)
    win_bound = (1, 100)
    bounds = [(sigma_bound[0], sigma_bound[1]),
              (win_bound[0], win_bound[1])]
    results_win_stay = fit_model_with_cv(model=win_stay_lose_shift,
                                         args=(confidence, feedback, n_trials),
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

    # Rescorla-Wagner model
    alpha_bound = (0, 1)
    sigma_bound = (1, 100)
    bias_bound = (0, 100)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]
    results_rw_symm = fit_model_with_cv(model=rw_symmetric_LR,
                                        args=(confidence, feedback, n_trials),
                                        bounds=bounds, n_trials=n_trials,
                                        start_value_number=50,
                                        solver="L-BFGS-B")
    alpha_array_rw_symm[session_idx] = results_rw_symm[0]
    sigma_array_rw_symm[session_idx] = results_rw_symm[1]
    bias_array_rw_symm[session_idx] = results_rw_symm[2]
    nll_array_rw_symm[session_idx] = results_rw_symm[3]
    aic_array_rw_symm[session_idx] = results_rw_symm[4]
    bic_array_rw_symm[session_idx] = results_rw_symm[5]
    pseudo_r2_array_rw_symm[session_idx] = results_rw_symm[6]

    # Choice Kernel model
    alpha_bound = (0, 1)
    sigma_bound = (1, 100)
    bias_bound = (0, 100)
    beta_bound = (0, 5)
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
    alpha_array_ck[session_idx] = results_ck[0]
    sigma_array_ck[session_idx] = results_ck[1]
    bias_array_ck[session_idx] = results_ck[2]
    beta_array_ck[session_idx] = results_ck[3]
    nll_array_ck[session_idx] = results_ck[4]
    aic_array_ck[session_idx] = results_ck[5]
    bic_array_ck[session_idx] = results_ck[6]
    pseudo_r2_array_ck[session_idx] = results_ck[7]

    # RW + Choice Kernel model
    alpha_bound = (0, 1)
    alpha_ck_bound = (0, 1)
    sigma_bound = (1, 100)
    bias_bound = (0, 100)
    beta_bound = (0, 5)
    beta_ck_bound = (0, 5)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (alpha_ck_bound[0], alpha_ck_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1]),
              (beta_ck_bound[0], beta_ck_bound[1])]
    results_rwck = fit_model_with_cv(model=RW_choice_kernel,
                                     args=(feedback, confidence, n_trials),
                                     bounds=bounds,
                                     n_trials=n_trials,
                                     start_value_number=50,
                                     solver="L-BFGS-B")
    alpha_array_rwck[session_idx] = results_rwck[0]
    alpha_ck_array_rwck[session_idx] = results_rwck[1]
    sigma_array_rwck[session_idx] = results_rwck[2]
    bias_array_rwck[session_idx] = results_rwck[3]
    beta_array_rwck[session_idx] = results_rwck[4]
    ck_beta_array_rwck[session_idx] = results_rwck[5]
    nll_array_rwck[session_idx] = results_rwck[6]
    aic_array_rwck[session_idx] = results_rwck[7]
    bic_array_rwck[session_idx] = results_rwck[8]
    pseudo_r2_array_rwck[session_idx] = results_rwck[9]

    # Rescorla-Wagner Performance Delta model
    alpha_bound = (0, 1)
    sigma_bound = (1, 100)
    bias_bound = (0, 10)
    w_rw_bound = (0, 10)
    w_delta_p_bound = (0, 100)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (w_rw_bound[0], w_rw_bound[1]),
              (w_delta_p_bound[0], w_delta_p_bound[1])]
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

    # Store session metrics
    session_metrics = {
        'pid': participant,
        'session': session,
        'condition': condition,
        'ntrials': n_trials,
        'nll_array_random_p': nll_array_random,
        'aic_array_random_p': aic_array_random,
        'bic_array_random_p': bic_array_random,
        'pseudo_r2_array_random_p': pseudo_r2_array_random,
        'mean_array_bias_p': mean_array_bias,
        'sd_array_bias_p': sd_array_bias,
        'nll_array_bias_p': nll_array_bias,
        'aic_array_bias_p': aic_array_bias,
        'bic_array_bias_p': bic_array_bias,
        'pseudo_r2_array_bias_p': pseudo_r2_array_bias,
        'std_WSLS_array_p': std_WSLS_array,
        'win_boundary_WSLS_array_p': win_boundary_WSLS_array,
        'nll_array_win_stay_p': nll_array_win_stay,
        'aic_array_win_stay_p': aic_array_win_stay,
        'bic_array_win_stay_p': bic_array_win_stay,
        'pseudo_r2_array_win_stay_p': pseudo_r2_array_win_stay,
        'alpha_array_rw_symm_p': alpha_array_rw_symm,
        'sigma_array_rw_symm_p': sigma_array_rw_symm,
        'bias_array_rw_symm_p': bias_array_rw_symm,
        'nll_array_rw_symm_p': nll_array_rw_symm,
        'aic_array_rw_symm_p': aic_array_rw_symm,
        'bic_array_rw_symm_p': bic_array_rw_symm,
        'pseudo_r2_array_rw_symm_p': pseudo_r2_array_rw_symm,
        'alpha_array_ck_p': alpha_array_ck,
        'sigma_array_ck_p': sigma_array_ck,
        'bias_array_ck_p': bias_array_ck,
        'beta_array_ck_p': beta_array_ck,
        'nll_array_ck_p': nll_array_ck,
        'aic_array_ck_p': aic_array_ck,
        'bic_array_ck_p': bic_array_ck,
        'pseudo_r2_array_ck_p': pseudo_r2_array_ck,
        'alpha_array_rwck_p': alpha_array_rwck,
        'sigma_array_rwck_p': sigma_array_rwck,
        'bias_array_rwck_p': bias_array_rwck,
        'beta_array_rwck_p': beta_array_rwck,
        'ck_beta_array_rwck_p': ck_beta_array_rwck,
        'nll_array_rwck_p': nll_array_rwck,
        'aic_array_rwck_p': aic_array_rwck,
        'bic_array_rwck_p': bic_array_rwck,
        'pseudo_r2_array_rwck_p': pseudo_r2_array_rwck,
        'alpha_array_delta_p_rw_p': alpha_array_delta_p_rw,
        'sigma_array_delta_p_rw_p': sigma_array_delta_p_rw,
        'nll_array_delta_p_rw_p': nll_array_delta_p_rw,
        'aic_array_delta_p_rw_p': aic_array_delta_p_rw,
        'bic_array_delta_p_rw_p': bic_array_delta_p_rw,
        'pseudo_r2_array_delta_p_rw_p': pseudo_r2_array_delta_p_rw,
    }

    return pid_session_tuple, session_metrics

def main(df):

    # Unique pid and sessions pairs
    unique_pairs = df[['pid', 'session']].drop_duplicates()

    # Convert the DataFrame to a list of tuples
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

    # index sessions
    unique_pairs_list = unique_pairs_list[:2]

    # Total number of sessions
    num_sessions = len(unique_pairs_list)
    print(f"Number of Sessions: {num_sessions}")

    # Initialize result dictionaries for all participants
    results = {
        'pid': np.empty(num_sessions, dtype=object),  # Use dtype=object for strings
        'session': np.zeros(num_sessions),
        'condition':  np.empty(num_sessions, dtype=object),  # Use dtype=object for strings
        'n_trials': np.zeros(num_sessions),
        'nll_array_random_p': np.zeros(num_sessions),
        'aic_array_random_p': np.zeros(num_sessions),
        'bic_array_random_p': np.zeros(num_sessions),
        'pseudo_r2_array_random_p': np.zeros(num_sessions),
        'mean_array_bias_p': np.zeros(num_sessions),
        'sd_array_bias_p': np.zeros(num_sessions),
        'nll_array_bias_p': np.zeros(num_sessions),
        'aic_array_bias_p': np.zeros(num_sessions),
        'bic_array_bias_p': np.zeros(num_sessions),
        'pseudo_r2_array_bias_p': np.zeros(num_sessions),
        'std_WSLS_array_p': np.zeros(num_sessions),
        'win_boundary_WSLS_array_p': np.zeros(num_sessions),
        'nll_array_win_stay_p': np.zeros(num_sessions),
        'aic_array_win_stay_p': np.zeros(num_sessions),
        'bic_array_win_stay_p': np.zeros(num_sessions),
        'pseudo_r2_array_win_stay_p': np.zeros(num_sessions),
        'alpha_array_rw_symm_p': np.zeros(num_sessions),
        'sigma_array_rw_symm_p': np.zeros(num_sessions),
        'bias_array_rw_symm_p': np.zeros(num_sessions),
        'nll_array_rw_symm_p': np.zeros(num_sessions),
        'aic_array_rw_symm_p': np.zeros(num_sessions),
        'bic_array_rw_symm_p': np.zeros(num_sessions),
        'pseudo_r2_array_rw_symm_p': np.zeros(num_sessions),
        'alpha_array_ck_p': np.zeros(num_sessions),
        'sigma_array_ck_p': np.zeros(num_sessions),
        'bias_array_ck_p': np.zeros(num_sessions),
        'beta_array_ck_p': np.zeros(num_sessions),
        'nll_array_ck_p': np.zeros(num_sessions),
        'aic_array_ck_p': np.zeros(num_sessions),
        'bic_array_ck_p': np.zeros(num_sessions),
        'pseudo_r2_array_ck_p': np.zeros(num_sessions),
        'alpha_array_rwck_p': np.zeros(num_sessions),
        'sigma_array_rwck_p': np.zeros(num_sessions),
        'bias_array_rwck_p': np.zeros(num_sessions),
        'beta_array_rwck_p': np.zeros(num_sessions),
        'ck_beta_array_rwck_p': np.zeros(num_sessions),
        'nll_array_rwck_p': np.zeros(num_sessions),
        'aic_array_rwck_p': np.zeros(num_sessions),
        'bic_array_rwck_p': np.zeros(num_sessions),
        'pseudo_r2_array_rwck_p': np.zeros(num_sessions),
        'alpha_array_delta_p_rw_p': np.zeros(num_sessions),
        'sigma_array_delta_p_rw_p': np.zeros(num_sessions),
        'nll_array_delta_p_rw_p': np.zeros(num_sessions),
        'aic_array_delta_p_rw_p': np.zeros(num_sessions),
        'bic_array_delta_p_rw_p': np.zeros(num_sessions),
        'pseudo_r2_array_delta_p_rw_p': np.zeros(num_sessions),
    }

    # Use multiprocessing to process each participant in parallel
    with Pool(2) as pool:
        for session_id, metrics in tqdm(pool.starmap(process_session,
                                                      [(pid_session_tuple, df)
                                                       for pid_session_tuple
                                                       in unique_pairs_list]),
                                         total=num_sessions):

            # Index session and save to that index within the session array
            print(unique_pairs_list, session_id)
            idx = unique_pairs_list.index(session_id)  # Find the index using list's index method
            for key in metrics:
                results[key][idx] = metrics[key]

    # Save model metrics array to excel file
    local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
    working_dir = r'metacognition-learning\comparative_models'
    save_path = r'results\fixed_feedback\model_comparison'
    name = 'model_metrics_sessions_CV.xlsx'
    save_path_full = os.path.join(local_folder,
                                  working_dir,
                                  save_path,
                                  name)

    df_m = pd.DataFrame(results)
    df_m.to_excel(save_path_full)

if __name__ == '__main__':
    # Import data - Fixed feedback condition (Experiment 1)
    current_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_directory)
    grandparent_directory = os.path.dirname(parent_directory)
    project_path = grandparent_directory
    fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
    data_file = r'main-20-12-14-processed_filtered.csv'
    full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
    df = pd.read_csv(full_path, low_memory=False)

    # Add session column
    df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

    # Run main
    main(df)


