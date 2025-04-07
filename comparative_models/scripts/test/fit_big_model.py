# -*- coding: utf-8 -*-
"""
Created on Thu May  2 18:12:05 2024

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
                        big_rw,
                        big_rw_trial,
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

# Metrics for big rw model
alpha_plus_array_big_rw_p = np.zeros(num_participants)
alpha_minus_array_big_rw_p = np.zeros(num_participants)
sigma_array_big_rw_p = np.zeros(num_participants)
mean_belief_array_big_rw_p = np.zeros(num_participants)
std_p_belief_array = np.zeros(num_participants)
w_array = np.zeros(num_participants)
util_func_bias_array = np.zeros(num_participants)
scaling_factor_array = np.zeros(num_participants)

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

    # Check if neut present
    any_neut_conditions = (df_p['condition'] == 'neut').any()
    if any_neut_conditions:
        pass
    else:
        continue

    # Session index
    session_idx = 0

    # Initialize empty lists to store session-wise data
    all_confidence = []
    all_feedback = []
    all_n_trials = []
    all_condition = []
    all_n_dots = []
    all_choice = []

    # Loop over sessions
    for session in df_p['session'].unique():

        # Get current session data, one row per trial
        df_s = df_p[df_p['session'] == session]

        if (df_s.condition == 'neut').any():
            pass
        else:
            continue

        # Only feedback trials
        df_s = df_s[df_s['condition'] != 'baseline']

        # N dots per subtrial
        n_dots = df_s.correct.values
        all_n_dots.extend(n_dots)

        # Choice
        all_choice.extend(df_s.estimate.values)

        # One row per trial - take last subtrial stats
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        # Get variables
        confidence = df_s['confidence'].values
        feedback = df_s['feedback'].values
        n_trials = len(df_s)

        condition = df_s['condition'].values
        n_dots = df_s.correct.values

        # Append session data to the lists
        all_confidence.extend(confidence)
        all_feedback.extend(feedback)
        all_n_trials.append(n_trials)  # This is a single value per session, so we append directly
        all_condition.extend(condition)

    # Convert lists to arrays (if needed)
    confidence = np.array(all_confidence)
    feedback = np.array(all_feedback)
    n_trials = np.sum(np.array(all_n_trials))
    condition = np.array(all_condition)
    n_dots = np.array(all_n_dots)
    choices = np.array(all_choice)

    if n_trials < 20:
        continue

    # Big RW model

    # Set bounds
    alpha_plus_bound = (0, 1)      # Alpha_plus
    alpha_minus_bound = (0, 1)      # Alpha_minus
    sigma_bound = (1, 10)    # Standard deviation
    mean_p_belief_bound = (20, 150)
    std_p_belief_bound = (0, 100)
    w_bound = (0, 1)
    util_func_bias_bound = (-100, 100)
    scaling_bound = (0.001, 100)
    #bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_plus_bound[0], alpha_plus_bound[1]),
              (alpha_minus_bound[0], alpha_minus_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (mean_p_belief_bound[0], mean_p_belief_bound[1]),
              (std_p_belief_bound[0], std_p_belief_bound[1]),
              (w_bound[0], w_bound[1]),
              (util_func_bias_bound[0], util_func_bias_bound[1]),
              (scaling_bound[0], scaling_bound[1]),
              ]

    # Get results
    results_big_rw = fit_model(model=big_rw,
                                args=(n_trials,
                                      n_dots,
                                      0, #min_value,
                                      100, #max_value,
                                      confidence,
                                      feedback),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=start_value_number,
                                solver="L-BFGS-B",
                                )

    best_alpha_plus = results_big_rw[0]
    best_alpha_minus = results_big_rw[1]
    best_std = results_big_rw[2]
    best_mean_p_belief = results_big_rw[3]
    best_std_p_belief = results_big_rw[4]
    best_w = results_big_rw[5]
    best_util_func_bias = results_big_rw[6]
    best_scaling_factor = results_big_rw[7]
    nll = results_big_rw[8]
    aic = results_big_rw[9]
    bic = results_big_rw[10]
    pseudo_r2 = results_big_rw[11]

    alpha_plus_array_big_rw_p[participant_idx] = best_alpha_plus
    alpha_minus_array_big_rw_p[participant_idx] = best_alpha_minus
    sigma_array_big_rw_p[participant_idx] = best_std
    mean_belief_array_big_rw_p[participant_idx]  = best_mean_p_belief
    std_p_belief_array[participant_idx]  = best_std_p_belief
    w_array[participant_idx] = best_w
    util_func_bias_array[participant_idx] = best_util_func_bias
    scaling_factor_array[participant_idx] = best_scaling_factor
    nll_array_big_rw_p[participant_idx] = nll
    aic_array_big_rw_p[participant_idx] = aic
    bic_array_big_rw_p[participant_idx] = bic

    # Increment participant index
    participant_idx += 1

    # Save model metrics to excel file
    local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
    working_dir = r'metacognition-learning\comparative_models'
    save_path = r'results\Fixed_feedback\model_comparison\Concat'
    name = 'big_rw_metrics.xlsx'
    save_path_full = os.path.join(local_folder, working_dir, save_path, name)
    model_metric_dict = {
        'alpha_plus_array_big_rw_p': alpha_plus_array_big_rw_p,
        'alpha_minus_array_big_rw_p': alpha_minus_array_big_rw_p,
        'sigma_array_big_rw_p': sigma_array_big_rw_p,
        'mean_belief_array_big_rw_p': mean_belief_array_big_rw_p,
        'std_p_belief_array': std_p_belief_array,
        'w_array': w_array,
        'util_func_bias_array': util_func_bias_array,
        'scaling_array': scaling_factor_array,
        'nll_array_big_rw_p': nll_array_big_rw_p,
        'aic_array_big_rw_p': aic_array_big_rw_p,
        'bic_array_big_rw_p': bic_array_big_rw_p,
        'pid': participant,
        'bdi': df_p.bdi.dropna().unique()[0]
    }
    df_m = pd.DataFrame(model_metric_dict)
    df_m.to_excel(save_path_full)


#%  plot prediction vs true

    participant_choice = choices
    participant_confidence = confidence


    x = [best_alpha_plus, best_alpha_minus, best_std,
         best_mean_p_belief, best_std_p_belief, best_w, best_util_func_bias,
         best_scaling_factor]
    [nlls, sigma_vec, choice_pred,
     confidence_pred, fb_hist_belief] = big_rw_trial(x,
                                                     n_trials,
                                                     n_dots,
                                                     0, # min_value,
                                                     100, # max_value,
                                                     confidence,
                                                     feedback)

    # Plot settings
    plt.figure(figsize=(14, 6))

    # Plot for Participant Choices vs. Model Predictions
    ax = plt.subplot(1, 2, 1)  # This creates a subplot (1 row, 2 columns, 1st subplot)
    plt.plot(participant_choice, label='Participant Choice', color='blue', marker='o')
    plt.plot(choice_pred, label='Model Prediction', color='red', linestyle='--')
    plt.title('Participant Choice vs. Model Prediction')
    plt.xlabel('Trial Number')
    plt.ylabel('Choice')
    plt.legend()

    # Plot for Participant Confidence vs. Model Predictions
    ax2 = plt.subplot(1, 2, 2)  # This creates a subplot (1 row, 2 columns, 2nd subplot)
    plt.plot(participant_confidence, label='Participant Confidence', color='green', marker='o')
    plt.plot(confidence_pred, label='Model Prediction', color='purple', linestyle='--')
    plt.fill_between(range(len(confidence_pred)),
                     confidence_pred - best_std,
                     confidence_pred + best_std, color='purple',
                     alpha=0.1, label='1 STD')
    plt.title('Participant Confidence vs. Model Prediction')
    plt.xlabel('Trial Number')
    plt.ylabel('Confidence')
    plt.ylim(0,100)
    plt.legend()

    for axi in [ax, ax2]:
        axi.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.show()

