# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:56:27 2024

@author: carll
"""


# Analysing data - Varied feedback - condition - concateneded sessions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from src.utility_functions import (add_session_column,
                                   write_metrics_to_excel,
                                   read_metrics_from_excel,
                                   assign_metrics_from_dict
                                   )
from src.models import (fit_model,
                        fit_random_model,
                        random_model_w_bias,
                        win_stay_lose_shift,
                        rw_static,
                        rw_symmetric_LR,
                        rw_cond_LR,
                        choice_kernel,
                        RW_choice_kernel,
                        delta_P_RW,
                        big_rw
                        )

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
#%%
df = pd.read_csv(full_path, low_memory=False)

# Add session column
df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Number of participants and maximum number of sessions per participant
num_participants = len(df.pid.unique())
max_sessions = df.groupby('pid')['session'].nunique().max()

# Set model independent parameters
start_value_number = 50 # n random initiations during fit procedure

# Pre-allocate arrays for participant-level metrics
# Metrics for Random Model
nll_array_random_p = np.zeros(num_participants)
aic_array_random_p = np.zeros(num_participants)
bic_array_random_p = np.zeros(num_participants)

# Metrics for Bias Model
nll_array_bias_p = np.zeros(num_participants)
aic_array_bias_p = np.zeros(num_participants)
bic_array_bias_p = np.zeros(num_participants)

# Metrics for Win-Stay-Lose-Shift Model
nll_array_win_stay_p = np.zeros(num_participants)
aic_array_win_stay_p = np.zeros(num_participants)
bic_array_win_stay_p = np.zeros(num_participants)

# Metrics for RW Static Model
alpha_array_rw_static_p = np.zeros(num_participants)
sigma_array_rw_static_p = np.zeros(num_participants)
nll_array_rw_static_p = np.zeros(num_participants)
aic_array_rw_static_p = np.zeros(num_participants)
bic_array_rw_static_p = np.zeros(num_participants)

# Metrics for RW Symmetric Model
alpha_array_rw_symm_p = np.zeros(num_participants)
sigma_array_rw_symm_p = np.zeros(num_participants)
nll_array_rw_symm_p = np.zeros(num_participants)
aic_array_rw_symm_p = np.zeros(num_participants)
bic_array_rw_symm_p = np.zeros(num_participants)

# Metrics for RW Cond Model
alpha_neut_array_rw_cond_p = np.zeros(num_participants)
alpha_pos_array_rw_cond_p = np.zeros(num_participants)
alpha_neg_array_rw_cond_p = np.zeros(num_participants)
sigma_array_rw_cond_p = np.zeros(num_participants)
nll_array_rw_cond_p = np.zeros(num_participants)
aic_array_rw_cond_p = np.zeros(num_participants)
bic_array_rw_cond_p = np.zeros(num_participants)

# Metrics for choice kernel
alpha_array_ck_p = np.zeros(num_participants)
sigma_array_ck_p = np.zeros(num_participants)
nll_array_ck_p = np.zeros(num_participants)
aic_array_ck_p = np.zeros(num_participants)
bic_array_ck_p = np.zeros(num_participants)

# Metrics for RW + choice kernel
alpha_array_rwck_p = np.zeros(num_participants)
sigma_array_rwck_p = np.zeros(num_participants)
nll_array_rwck_p = np.zeros(num_participants)
aic_array_rwck_p = np.zeros(num_participants)
bic_array_rwck_p = np.zeros(num_participants)

# Metrics for Delta P RW Model
alpha_array_delta_p_rw_p = np.zeros(num_participants)
sigma_array_delta_p_rw_p = np.zeros(num_participants)
nll_array_delta_p_rw_p = np.zeros(num_participants)
aic_array_delta_p_rw_p = np.zeros(num_participants)
bic_array_delta_p_rw_p = np.zeros(num_participants)

# Metrics for big rw model
alpha_plus_array_big_rw_p = np.zeros(num_participants)
alpha_minus_array_big_rw_p = np.zeros(num_participants)
sigma_array_big_rw_p = np.zeros(num_participants)
nll_array_big_rw_p = np.zeros(num_participants)
aic_array_big_rw_p = np.zeros(num_participants)
bic_array_big_rw_p = np.zeros(num_participants)


# Participant index
participant_idx = 0

# Loop over participants
for participant in tqdm(df.pid.unique(),
                        total=len(df.pid.unique()),
                        desc='Participant Loop'):

    # Current participant data only
    df_p = df[df['pid'] == participant]

    # Session index
    session_idx = 0

    # Initialize empty lists to store session-wise data
    all_confidence = []
    all_feedback = []
    all_n_trials = []
    all_condition = []
    all_performance = []

    # Loop over sessions
    for session in df_p['session'].unique():
        # Get current session data, one row per trial
        df_s = df_p[df_p['session'] == session]

        # Only feedback trials
        df_s = df_s[df_s['condition'] != 'baseline']

        # Calculate the difference between 'estimate' and 'correct'
        df_s['difference'] = df_s['estimate'] - df_s['correct']

        # Group by 'trial' and calculate the mean of the differences
        error_avg = df_s.groupby('trial')['difference'].mean()

        # One row per trial
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        # Get variables
        confidence = df_s['confidence'].values
        feedback = df_s['feedback'].values
        n_trials = len(df_s)
        condition = df_s['condition'].values
        performance = -error_avg.values

        # Append session data to the lists
        all_confidence.extend(confidence)
        all_feedback.extend(feedback)
        all_n_trials.append(n_trials)  # This is a single value per session, so we append directly
        all_condition.extend(condition)
        all_performance.extend(performance)

    # Convert lists to arrays (if needed)
    confidence = np.array(all_confidence)
    feedback = np.array(all_feedback)
    n_trials = np.sum(np.array(all_n_trials))
    condition = np.array(all_condition)
    performance = np.array(all_performance)



    # Big RW model

    # Set bounds
    alpha_plus_bound = (0, 1)      # Alpha_plus
    alpha_minus_bound = (0, 1)      # Alpha_minus
    sigma_bound = (1, 100)    # Standard deviation
    #bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_plus_bound[0], alpha_plus_bound[1]),
              (alpha_minus_bound[0], alpha_minus_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              ]

    # Get results
    results_rw_static = fit_model(model=big_rw,
                                args=(confidence,
                                      feedback,
                                      n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=start_value_number,
                                solver="L-BFGS-B",
                                )

    best_alpha_plus = results_rw_static[0]
    best_alpha_minus = results_rw_static[1]
    best_std = results_rw_static[2]
    nll = results_rw_static[3]
    aic = results_rw_static[4]
    bic = results_rw_static[5]
    pseudo_r2 = results_rw_static[6]

    alpha_plus_array_big_rw_p[participant_idx] = best_alpha_plus
    alpha_minus_array_big_rw_p[participant_idx] = best_alpha_minus
    sigma_array_big_rw_p[participant_idx] = best_std
    nll_array_big_rw_p[participant_idx] = nll
    aic_array_big_rw_p[participant_idx] = aic
    bic_array_big_rw_p[participant_idx] = bic


    # Random confidence model
    # print('random model')
    nll_random_model = fit_random_model(prediction_range=100,
                                        n_trials=n_trials)

    # Get BIC and AIC
    k = 0
    random_model_aic = 2*k + 2*nll_random_model
    random_model_bic = k*np.log(n_trials) + 2*nll_random_model

    nll_array_random_p[participant_idx] = nll_random_model
    aic_array_random_p[participant_idx] = random_model_aic
    bic_array_random_p[participant_idx] = random_model_bic

    # Biased confidence model
    # print('biased model...')
    # Set Bounds
    mean_bound = (0, 100)   # Mean
    sigma_bound = (1, 100)  # Standard deviation
    bounds = [(mean_bound[0], mean_bound[1]),
              (sigma_bound[0],  sigma_bound[1]),
              ]

    # Get results
    results = fit_model(model=random_model_w_bias,
                        args=(all_confidence,
                              n_trials),
                        bounds=bounds,
                        n_trials=n_trials,
                        start_value_number=start_value_number,
                        solver="L-BFGS-B")

    best_mean_bias = results[0]
    best_std_bias = results[1]
    nll = results[2]
    aic = results[3]
    bic = results[4]
    pseudo_r2 = results[5]

    nll_array_bias_p[participant_idx] = nll
    aic_array_bias_p[participant_idx] = aic
    bic_array_bias_p[participant_idx] = bic

    # Get NLL from biased params
    NLL_biased_w_best_params = random_model_w_bias([best_mean_bias,
                                                    best_std_bias],
                                               confidence,
                                               n_trials,
                                                  )

    # Win-stay-lose-shift-model
    # print('WSLS model...')
    # Set Bounds
    sigma_bound = (1, 100)  # Standard deviation
    win_bound = (1, 100)    # Win boundary
    bounds = [(sigma_bound[0],  sigma_bound[1]),
              (win_bound[0],  win_bound[1])]

    # Get results
    results_win_stay = fit_model(model=win_stay_lose_shift,
                                 args=(confidence,
                                       feedback,
                                       n_trials
                                       ),
                                 bounds=bounds,
                                 n_trials=n_trials,
                                 start_value_number=start_value_number,
                                 solver="L-BFGS-B")

    best_std_WSLS = results_win_stay[0]
    best_win_boundary_WSLS = results_win_stay[1]


    nll_array_win_stay_p[participant_idx] = results_win_stay[2]
    aic_array_win_stay_p[participant_idx] = results_win_stay[3]
    bic_array_win_stay_p[participant_idx] = results_win_stay[4]


    # Rescorla wagner model - static - always update to previous confidence

    # Set bounds
    alpha_bound = (0, 1)      # Alpha
    sigma_bound = (1, 100)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]

    # Get results
    results_rw_static = fit_model(model=rw_static,
                                args=(confidence,
                                      feedback,
                                      n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=start_value_number,
                                solver="L-BFGS-B",
                                )

    best_alpha = results_rw_static[0]
    best_std = results_rw_static[1]
    best_bias = results_rw_static[2]
    nll = results_rw_static[3]
    aic = results_rw_static[4]
    bic = results_rw_static[5]
    pseudo_r2 = results_rw_static[6]

    alpha_array_rw_static_p[participant_idx] = best_alpha
    sigma_array_rw_static_p[participant_idx] = best_std
    nll_array_rw_static_p[participant_idx] = nll
    aic_array_rw_static_p[participant_idx] = aic
    bic_array_rw_static_p[participant_idx] = bic


    # Rescorla wagner model - continous

    # Set bounds
    alpha_bound = (0, 1)      # Alpha
    sigma_bound = (1, 100)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]

    # Get results
    results_rw_symm = fit_model(model=rw_symmetric_LR,
                                args=(confidence,
                                      feedback,
                                      n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=start_value_number,
                                solver="L-BFGS-B",
                                bias_model_best_params=False
                                )

    best_alpha = results_rw_symm[0]
    best_std = results_rw_symm[1]
    best_bias = results_rw_symm[2]
    nll = results_rw_symm[3]
    aic = results_rw_symm[4]
    bic = results_rw_symm[5]
    pseudo_r2 = results_rw_symm[6]

    alpha_array_rw_symm_p[participant_idx] = best_alpha
    sigma_array_rw_symm_p[participant_idx] = best_std
    nll_array_rw_symm_p[participant_idx] = nll
    aic_array_rw_symm_p[participant_idx] = aic
    bic_array_rw_symm_p[participant_idx] = bic


    # Rescorla wagner Condition alpha model

    # Set bounds

    alpha_neut_bound = (0, 1)  # Alpha neut
    alpha_pos_bound =  (0, 1)  # Alpha pos
    alpha_neg_bound =  (0, 1)  # Alpha neg
    sigma_bound = (1, 100)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_neut_bound[0], alpha_neut_bound[1]),
              (alpha_pos_bound[0], alpha_pos_bound[1]),
              (alpha_neg_bound[0], alpha_neg_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]

    # Get results
    results_rw_cond = fit_model(model=rw_cond_LR,
                                args=(confidence,
                                      feedback,
                                      n_trials,
                                      condition),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=start_value_number,
                                solver="L-BFGS-B",
                                bias_model_best_params=False)

    best_alpha_neut = results_rw_cond[0]
    best_alpha_pos = results_rw_cond[1]
    best_alpha_neg = results_rw_cond[2]
    best_std = results_rw_cond[3]
    best_bias = results_rw_cond[4]
    nll = results_rw_cond[5]
    aic = results_rw_cond[6]
    bic = results_rw_cond[7]
    pseudo_r2 = results_rw_cond[8]

    # RW_cond model
    alpha_neut_array_rw_cond_p[participant_idx] = best_alpha_neut
    alpha_pos_array_rw_cond_p[participant_idx] = best_alpha_pos
    alpha_neg_array_rw_cond_p[participant_idx] = best_alpha_neg
    sigma_array_rw_cond_p[participant_idx] = best_std
    nll_array_rw_cond_p[participant_idx] = nll
    aic_array_rw_cond_p[participant_idx] = aic
    bic_array_rw_cond_p[participant_idx] = bic

    # Choice Kernel Model

    # Set bounds
    alpha_bound = (0, 1)  # Alpha
    sigma_bound = (1, 100)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    beta_bound = (0, 5)        # Beta
    bounds = [
              (alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1])
              ]

    # Get results
    results_ck = fit_model(model=choice_kernel,
                                args=(confidence,
                                      n_trials,
                                      ),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=start_value_number,
                                solver="L-BFGS-B",
                                bias_model_best_params=False,

                                )

    best_alpha = results_ck[0]
    best_std = results_ck[1]
    best_bias = results_ck[2]
    best_beta = results_ck[3]
    nll = results_ck[4]
    aic = results_ck[5]
    bic = results_ck[6]
    pseudo_r2 = results_ck[7]

    # Choice Kernel model
    alpha_array_ck_p[participant_idx] = best_alpha
    sigma_array_ck_p[participant_idx] = best_std
    nll_array_ck_p[participant_idx] = nll
    aic_array_ck_p[participant_idx] = aic
    bic_array_ck_p[participant_idx] = bic


    # RW + Choice Kernel Model

    # Set bounds
    alpha_bound = (0, 1)  # Alpha
    alpha_ck_bound = (0, 1)  # Alpha choice kernel
    sigma_bound = (1, 100)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    beta_bound = (0, 5)        # Beta
    bounds = [
              (alpha_bound[0], alpha_bound[1]),
              (alpha_ck_bound[0], alpha_ck_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1])
              ]

    # Get results
    results_rwck = fit_model(model=RW_choice_kernel,
                             args=(feedback,
                                   confidence,
                                   n_trials,
                                   ),
                             bounds=bounds,
                             n_trials=n_trials,
                             start_value_number=start_value_number,
                             solver="L-BFGS-B",
                                bias_model_best_params=False)

    best_alpha = results_rwck[0]
    best_alpha_ck = results_rwck[1]
    best_std = results_rwck[2]
    best_bias = results_rwck[3]
    best_beta = results_rwck[4]
    nll = results_rwck[5]
    aic = results_rwck[6]
    bic = results_rwck[7]
    pseudo_r2 = results_rwck[8]

    # RW + Choice Kernel model
    alpha_array_rwck_p[participant_idx] = best_alpha
    sigma_array_rwck_p[participant_idx] = best_std
    nll_array_rwck_p[participant_idx] = nll
    aic_array_rwck_p[participant_idx] = aic
    bic_array_rwck_p[participant_idx] = bic


    # Delta P + Rescorla wagner model

    # Set bounds
    alpha_bound = (0, 1)    # Alpha
    sigma_bound = (1, 100)      # Standard deviation
    bias_bound = (0, 100)       # Mean at first trial
    w_rw_bound = (0, 10)       # Weight for RW update
    w_delta_p_bound = (0, 10)  # Weight for delta p
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (w_rw_bound[0], w_rw_bound[1]),
              (w_delta_p_bound[0], w_delta_p_bound[1])]

    # Get results
    results_delta_p_rw = fit_model(model=delta_P_RW,
                                   args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                   bounds=bounds,
                                   n_trials=n_trials,
                                   start_value_number=start_value_number,
                                   solver="L-BFGS-B",
                                   bias_model_best_params=False)

    best_alpha = results_delta_p_rw[0]
    best_std = results_delta_p_rw[1]
    best_bias = results_delta_p_rw[2]
    best_w_rw = results_delta_p_rw[3]
    best_w_delta_p = results_delta_p_rw[4]
    nll = results_delta_p_rw[5]
    aic = results_delta_p_rw[6]
    bic = results_delta_p_rw[7]
    pseudo_r2 = results_delta_p_rw[8]


    alpha_array_delta_p_rw_p[participant_idx] = best_alpha
    sigma_array_delta_p_rw_p[participant_idx] = best_std
                                                #best_bias
                                                #best_w_rw
                                                #best_w_delta_p
    nll_array_delta_p_rw_p[participant_idx] = nll
    aic_array_delta_p_rw_p[participant_idx] = aic
    bic_array_delta_p_rw_p[participant_idx] = bic

    # Increment participant index
    participant_idx += 1

    # Save model metrics to excel file
    local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
    working_dir = r'metacognition-learning\comparative_models'
    save_path = r'results\Fixed_feedback\model_comparison\Concat'
    name = 'model_metrics.xlsx'
    save_path_full = os.path.join(local_folder, working_dir, save_path, name)
    model_metric_dict = {
    # Metrics for Random Model
    'nll_array_random_p': nll_array_random_p,
    'aic_array_random_p': aic_array_random_p,
    'bic_array_random_p': bic_array_random_p,

    # Metrics for Bias Model
    'nll_array_bias_p': nll_array_bias_p,
    'aic_array_bias_p': aic_array_bias_p,
    'bic_array_bias_p': bic_array_bias_p,

    # Metrics for Win-Stay-Lose-Shift Model
    'nll_array_win_stay_p': nll_array_win_stay_p,
    'aic_array_win_stay_p': aic_array_win_stay_p,
    'bic_array_win_stay_p': bic_array_win_stay_p,

    # Metrics for RW Static Model
    'alpha_array_rw_static_p': alpha_array_rw_static_p,
    'sigma_array_rw_static_p': sigma_array_rw_static_p,
    'nll_array_rw_static_p': nll_array_rw_static_p,
    'aic_array_rw_static_p': aic_array_rw_static_p,
    'bic_array_rw_static_p': bic_array_rw_static_p,

    # Metrics for RW Symmetric Model
    'alpha_array_rw_symm_p': alpha_array_rw_symm_p,
    'sigma_array_rw_symm_p': sigma_array_rw_symm_p,
    'nll_array_rw_symm_p': nll_array_rw_symm_p,
    'aic_array_rw_symm_p': aic_array_rw_symm_p,
    'bic_array_rw_symm_p': bic_array_rw_symm_p,

    # Metrics for RW Cond Model
    'alpha_neut_array_rw_cond_p': alpha_neut_array_rw_cond_p,
    'alpha_pos_array_rw_cond_p': alpha_pos_array_rw_cond_p,
    'alpha_neg_array_rw_cond_p': alpha_neg_array_rw_cond_p,
    'sigma_array_rw_cond_p': sigma_array_rw_cond_p,
    'nll_array_rw_cond_p': nll_array_rw_cond_p,
    'aic_array_rw_cond_p': aic_array_rw_cond_p,
    'bic_array_rw_cond_p': bic_array_rw_cond_p,

    # Metrics for choice kernel
    'alpha_array_ck_p': alpha_array_ck_p,
    'sigma_array_ck_p': sigma_array_ck_p,
    'nll_array_ck_p': nll_array_ck_p,
    'aic_array_ck_p': aic_array_ck_p,
    'bic_array_ck_p': bic_array_ck_p,

    # Metrics for RW + choice kernel
    'alpha_array_rwck_p': alpha_array_rwck_p,
    'sigma_array_rwck_p': sigma_array_rwck_p,
    'nll_array_rwck_p': nll_array_rwck_p,
    'aic_array_rwck_p': aic_array_rwck_p,
    'bic_array_rwck_p': bic_array_rwck_p,

    # Metrics for Delta P RW Model
    'alpha_array_delta_p_rw_p': alpha_array_delta_p_rw_p,
    'sigma_array_delta_p_rw_p': sigma_array_delta_p_rw_p,
    'nll_array_delta_p_rw_p': nll_array_delta_p_rw_p,
    'aic_array_delta_p_rw_p': aic_array_delta_p_rw_p,
    'bic_array_delta_p_rw_p': bic_array_delta_p_rw_p,
    }
    df_m = pd.DataFrame(model_metric_dict)
    df_m.to_excel(save_path_full)

#%% Load data

# Read the metrics back from the Excel file and assign to variables
local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
working_dir = r'metacognition-learning\comparative_models'
save_path = r'results\Fixed_feedback\model_comparison\Concat'
name = 'model_metrics.xlsx'
save_path_full = os.path.join(local_folder, working_dir, save_path, name)
df_m = pd.read_excel(save_path_full)

#df_m = df_m.head(132)
# %% Get mean and sem

# Data for random model
random_model_mean_nll = np.mean(df_m.nll_array_random_p)
random_model_sem_nll = (np.std(df_m.nll_array_random_p) /
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

# Data for RW static LR model
rw_static_model_mean_nll = np.mean(nll_array_rw_static_p)
rw_static_model_sem_nll = np.std(nll_array_rw_static_p) / \
                          np.sqrt(len(nll_array_rw_static_p))
rw_static_model_mean_aic = np.mean(aic_array_rw_static_p)
rw_static_model_sem_aic = np.std(aic_array_rw_static_p) / \
                          np.sqrt(len(aic_array_rw_static_p))
rw_static_model_mean_bic = np.mean(bic_array_rw_static_p)
rw_static_model_sem_bic = np.std(bic_array_rw_static_p) / \
                          np.sqrt(len(bic_array_rw_static_p))

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

# Data for RW condition LR model
rw_cond_model_mean_nll = np.mean(nll_array_rw_cond_p)
rw_cond_model_sem_nll = np.std(nll_array_rw_cond_p) / \
                        np.sqrt(len(nll_array_rw_cond_p))
rw_cond_model_mean_aic = np.mean(aic_array_rw_cond_p)
rw_cond_model_sem_aic = np.std(aic_array_rw_cond_p) / \
                        np.sqrt(len(aic_array_rw_cond_p))
rw_cond_model_mean_bic = np.mean(bic_array_rw_cond_p)
rw_cond_model_sem_bic = np.std(bic_array_rw_cond_p) / \
                        np.sqrt(len(bic_array_rw_cond_p))

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
    ['rw_static', 'nll_array_rw_static_p', 'aic_array_rw_static_p', 'bic_array_rw_static_p'],
    ['rw_symm', 'nll_array_rw_symm_p', 'aic_array_rw_symm_p', 'bic_array_rw_symm_p'],
    ['rw_cond', 'nll_array_rw_cond_p', 'aic_array_rw_cond_p', 'bic_array_rw_cond_p'],
    ['ck', 'nll_array_ck_p', 'aic_array_ck_p', 'bic_array_ck_p'],
    ['rwck', 'nll_array_rwck_p', 'aic_array_rwck_p', 'bic_array_rwck_p'],
    ['delta_p_rw', 'nll_array_delta_p_rw_p', 'aic_array_delta_p_rw_p', 'bic_array_delta_p_rw_p'],
]

# Loop through each model and metric, calculate mean and SEM, and store in results dictionary
for model, nll_col, aic_col, bic_col in models_metrics:
    results[f'{model}_model_mean_nll'], results[f'{model}_model_sem_nll'] = calculate_mean_sem(df_m[nll_col])
    results[f'{model}_model_mean_aic'], results[f'{model}_model_sem_aic'] = calculate_mean_sem(df_m[aic_col])
    results[f'{model}_model_mean_bic'], results[f'{model}_model_sem_bic'] = calculate_mean_sem(df_m[bic_col])

# =============================================================================
#
# # Iterating over the results dictionary and dynamically creating variables
# for key, value in results.items():
#     exec(f"{key} = {value}")
#
# # At this point, variables like random_mean_nll, random_sem_nll, etc., are created and assigned.
# =============================================================================

#%% Metrics: Relative fit in relation to best model fit

import numpy as np
import pandas as pd

# Assuming df_m is your DataFrame containing the models' metrics for each participant

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_array_random_p', 'aic_array_random_p', 'bic_array_random_p'],
    ['bias', 'nll_array_bias_p', 'aic_array_bias_p', 'bic_array_bias_p'],
    ['win_stay', 'nll_array_win_stay_p', 'aic_array_win_stay_p', 'bic_array_win_stay_p'],
    ['rw_static', 'nll_array_rw_static_p', 'aic_array_rw_static_p', 'bic_array_rw_static_p'],
    ['rw_symm', 'nll_array_rw_symm_p', 'aic_array_rw_symm_p', 'bic_array_rw_symm_p'],
    ['rw_cond', 'nll_array_rw_cond_p', 'aic_array_rw_cond_p', 'bic_array_rw_cond_p'],
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

# Example of how you might use the results
print(results['random_model_mean_nll'], results['random_model_sem_nll'])
# Continue for other metrics and models as needed

#%% Set variables explicitly

random_model_mean_nll = results['random_model_mean_nll']
random_model_sem_nll = results['random_model_sem_nll']
random_model_mean_aic = results['random_model_mean_aic']
random_model_sem_aic = results['random_model_sem_aic']
random_model_mean_bic = results['random_model_mean_bic']
random_model_sem_bic = results['random_model_sem_bic']

bias_model_mean_nll = results['bias_model_mean_nll']
bias_model_sem_nll = results['bias_model_sem_nll']
bias_model_mean_aic = results['bias_model_mean_aic']
bias_model_sem_aic = results['bias_model_sem_aic']
bias_model_mean_bic = results['bias_model_mean_bic']
bias_model_sem_bic = results['bias_model_sem_bic']

win_stay_model_mean_nll = results['win_stay_model_mean_nll']
win_stay_model_sem_nll = results['win_stay_model_sem_nll']
win_stay_model_mean_aic = results['win_stay_model_mean_aic']
win_stay_model_sem_aic = results['win_stay_model_sem_aic']
win_stay_model_mean_bic = results['win_stay_model_mean_bic']
win_stay_model_sem_bic = results['win_stay_model_sem_bic']

rw_static_model_mean_nll = results['rw_static_model_mean_nll']
rw_static_model_sem_nll = results['rw_static_model_sem_nll']
rw_static_model_mean_aic = results['rw_static_model_mean_aic']
rw_static_model_sem_aic = results['rw_static_model_sem_aic']
rw_static_model_mean_bic = results['rw_static_model_mean_bic']
rw_static_model_sem_bic = results['rw_static_model_sem_bic']

rw_symm_model_mean_nll = results['rw_symm_model_mean_nll']
rw_symm_model_sem_nll = results['rw_symm_model_sem_nll']
rw_symm_model_mean_aic = results['rw_symm_model_mean_aic']
rw_symm_model_sem_aic = results['rw_symm_model_sem_aic']
rw_symm_model_mean_bic = results['rw_symm_model_mean_bic']
rw_symm_model_sem_bic = results['rw_symm_model_sem_bic']

rw_cond_model_mean_nll = results['rw_cond_model_mean_nll']
rw_cond_model_sem_nll = results['rw_cond_model_sem_nll']
rw_cond_model_mean_aic = results['rw_cond_model_mean_aic']
rw_cond_model_sem_aic = results['rw_cond_model_sem_aic']
rw_cond_model_mean_bic = results['rw_cond_model_mean_bic']
rw_cond_model_sem_bic = results['rw_cond_model_sem_bic']

ck_model_mean_nll = results['ck_model_mean_nll']
ck_model_sem_nll = results['ck_model_sem_nll']
ck_model_mean_aic = results['ck_model_mean_aic']
ck_model_sem_aic = results['ck_model_sem_aic']
ck_model_mean_bic = results['ck_model_mean_bic']
ck_model_sem_bic = results['ck_model_sem_bic']

rwck_model_mean_nll = results['rwck_model_mean_nll']
rwck_model_sem_nll = results['rwck_model_sem_nll']
rwck_model_mean_aic = results['rwck_model_mean_aic']
rwck_model_sem_aic = results['rwck_model_sem_aic']
rwck_model_mean_bic = results['rwck_model_mean_bic']
rwck_model_sem_bic = results['rwck_model_sem_bic']

delta_p_rw_model_mean_nll = results['delta_p_rw_model_mean_nll']
delta_p_rw_model_sem_nll = results['delta_p_rw_model_sem_nll']
delta_p_rw_model_mean_aic = results['delta_p_rw_model_mean_aic']
delta_p_rw_model_sem_aic = results['delta_p_rw_model_sem_aic']
delta_p_rw_model_mean_bic = results['delta_p_rw_model_mean_bic']
delta_p_rw_model_sem_bic = results['delta_p_rw_model_sem_bic']

#%%  Histogram of best model

metric = [
            [df_m.nll_array_random_p,
             df_m.nll_array_bias_p,
             df_m.nll_array_win_stay_p,
             df_m.nll_array_rw_symm_p,
             df_m.nll_array_rw_cond_p,
             df_m.nll_array_ck_p,
             df_m.nll_array_rwck_p,
             df_m.nll_array_delta_p_rw_p],

            [df_m.aic_array_random_p,
             df_m.aic_array_bias_p,
             df_m.aic_array_win_stay_p,
             df_m.aic_array_rw_symm_p,
             df_m.aic_array_rw_cond_p,
             df_m.aic_array_ck_p,
             df_m.aic_array_rwck_p,
             df_m.aic_array_delta_p_rw_p],

            [df_m.bic_array_random_p,
             df_m.bic_array_bias_p,
             df_m.bic_array_win_stay_p,
             df_m.bic_array_rw_symm_p,
             df_m.bic_array_rw_cond_p,
             df_m.bic_array_ck_p,
             df_m.bic_array_rwck_p,
             df_m.bic_array_delta_p_rw_p]

            ]


# Loop over metrics
fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize=(6,6))
for metric_list, metric_name, ax in zip(metric,
                                    ['NLL', 'AIC', 'BIC'],
                                    [ax, ax2, ax3]):

    score_board = []
    pids = []
    # Loop over each participant's model score
    for rand, bias, wsls, rw, rw_cond, ck, rwck, delta_p_rw, pid in zip(
                                                             metric_list[0],
                                                             metric_list[1],
                                                             metric_list[2],
                                                             metric_list[3],
                                                             metric_list[4],
                                                             metric_list[5],
                                                             metric_list[6],
                                                             metric_list[7],
                                                 range(len(metric_list[6]))):


        # Scores from different models
        scores = np.array([rand,
                           bias,
                           wsls,
                           rw,
                           rw_cond,
                           ck,
                           rwck,
                           delta_p_rw,
                           ])

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
        df_rwpd_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
        pid_rwpd_best = df_rwpd_fit[df_rwpd_fit.best_model==4].pid
    else:
        pass

    models = ['random', 'biased',
              'WSLS', 'RW',
              'RW_Cond','CK',
              'RWCK', 'RWPD']

    counts = [score_board.count(0),
              score_board.count(1),
              score_board.count(2),
              score_board.count(3),
              score_board.count(7),
              score_board.count(5),
              score_board.count(6),
              score_board.count(4),
              ]

    bar_colors = ['blue', 'green',
                  'red', 'purple',
                  'cyan', 'pink', 'orange', 'silver']
    bars = ax.bar(models, counts, color=bar_colors)

    # Customizing the Axes
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7
                   ])
    ax.set_ylabel('n participants')
    ax.set_xlim(-1, len(models))
    ax.set_title(metric_name)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig.suptitle('Best Model')

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = r'model_comparison_histogram_concat.png'
save_path = os.path.join(project_path, r'comparative_models',
                         result_path, file_name)

fig.savefig(save_path, dpi=1200, bbox_inches='tight')

plt.tight_layout()
plt.show()

# %% Plot mean across conditions

fig, ax = plt.subplots(figsize=(8, 4))

# Define x-coordinates and offsets for each model within the group
x = np.arange(0,9,3)  # Base x-coordinates for metrics
offset = 0.35 # Offset for each model within a group
capsize = 8

# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green',
          'red', 'purple',
          'cyan',
          'pink', 'orange',
          'silver']

# Plotting NLL for each model
model_means = [
    random_model_mean_nll, bias_model_mean_nll,
    win_stay_model_mean_nll, rw_symm_model_mean_nll,
    rw_cond_model_mean_nll,
    ck_model_mean_nll,
    rwck_model_mean_nll,
    delta_p_rw_model_mean_nll
]
model_sems = [
    random_model_sem_nll, bias_model_sem_nll,
    win_stay_model_sem_nll, rw_symm_model_sem_nll,
    rw_cond_model_sem_nll,
    ck_model_sem_nll,
    rwck_model_sem_nll,
    delta_p_rw_model_sem_nll
]
model_names = [
    "Random", "Biased",
    "Win-Stay-Lose-Shift",
    "RW Symmetric LR",
    "RW Condition LR",
    "Choice Kernel",
    "RW + Choice Kernel",
    "RW + Performance Delta"
]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(
        x[0] + offset * (i - 1.5), mean, yerr=sem,
        fmt='o', capsize=capsize, color=colors[i], label=model_names[i]
    )

# Plotting AIC for each model
model_means = [
    random_model_mean_aic, bias_model_mean_aic,
    win_stay_model_mean_aic, rw_symm_model_mean_aic,
    rw_cond_model_mean_aic,
    ck_model_mean_aic,
    rwck_model_mean_aic,
    delta_p_rw_model_mean_aic
]
model_sems = [
    random_model_sem_aic, bias_model_sem_aic,
    win_stay_model_sem_aic, rw_symm_model_sem_aic,
    rw_cond_model_sem_aic,
    ck_model_sem_aic,
    rwck_model_sem_aic,
    delta_p_rw_model_sem_aic
]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(
        x[1] + offset * (i - 1.5), mean, yerr=sem,
        fmt='o', capsize=capsize, color=colors[i]
    )

# Plotting BIC for each model
model_means = [
    random_model_mean_bic, bias_model_mean_bic,
    win_stay_model_mean_bic, rw_symm_model_mean_bic,
    rw_cond_model_mean_bic,
    ck_model_mean_bic,
    rwck_model_mean_bic,
    delta_p_rw_model_mean_bic

]
model_sems = [
    random_model_sem_bic, bias_model_sem_bic,
    win_stay_model_sem_bic, rw_symm_model_sem_bic,
    rw_cond_model_sem_bic,
    ck_model_sem_bic,
    rwck_model_sem_bic,
    delta_p_rw_model_sem_bic
]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(
        x[2] + offset * (i - 1.5), mean, yerr=sem,
        fmt='o', capsize=capsize, color=colors[i]
    )

# Customizing the Axes
ax.set_xticks(x)
ax.set_xticklabels(['NLL', 'AIC', 'BIC'])
ax.set_ylabel('Model fits')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adding a legend
ax.legend(model_names)

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = r'model_comparison_NLL_concat_means.svg'
save_path = os.path.join(project_path, r'comparative_models',
                         result_path, file_name)

fig.savefig(save_path, dpi=1200, bbox_inches = 'tight')

plt.tight_layout()
plt.show()

#%% Plot the alpha of RWPD vs BDI
import scipy.stats as stats
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
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)
# alpha_array_rw_symm_p
# alpha_neut_array_rw_cond_p
# alpha_pos_array_rw_cond_p
# alpha_neg_array_rw_cond_p
# alpha_array_delta_p_rw_p

x = df_m.alpha_neut_array_rw_cond_p#[pid_rwpd_best.values] # alpha_neut_array_rw_cond_p
x_idx = np.array([i for i,j in enumerate(x)]) # only above 0 learning rates
x = x.values[x_idx]
y = bdi#[pid_rwpd_best.values]
y = y[x_idx]

# Add a constant to the input data for the intercept
x_with_intercept = sm.add_constant(x)

# Perform the linear regression
model = sm.OLS(y, x_with_intercept)
results = model.fit()

# Getting the regression line
pred = results.predict(x_with_intercept)

# Summary of the model
print(results.summary())

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
ax.scatter(x, y, alpha=0.75, label='Data Points')
ax.plot(x, pred, ls='-', c='r', label=f'Linear Fit: y = {results.params[1]:.2f}x + {results.params[0]:.2f}')
ax.set_xlabel('Learning rate')
ax.set_ylabel('BDI score')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate p-value and other statistics
ax.text(0.60, 0.95, f'Slope: {results.params[1]:.2f}\nIntercept: {results.params[0]:.2f}\n'
                     f'R-squared: {results.rsquared:.3f}\nP-value: {results.pvalues[1]:.3g}',
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10,
        bbox=dict(boxstyle="round", alpha=0.9, color='w'))

#ax.legend()

plt.show()

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI_concat.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()
