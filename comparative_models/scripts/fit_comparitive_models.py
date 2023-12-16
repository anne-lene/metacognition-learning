# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:02:33 2023

@author: carll
"""

# Analysing data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from src.utility_functions import add_session_column
from src.models import (fit_model,
                        fit_random_model,
                        random_model_w_bias,
                        win_stay_lose_shift,
                        rw_symmetric_LR)

# Import data
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

# Number of participants and maximum number of sessions per participant
num_participants = len(df.pid.unique())
max_sessions = df.groupby('pid')['session'].nunique().max()

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

# Metrics for RW Symmetric Model
alpha_array_rw_symm_p = np.zeros(num_participants)
sigma_array_rw_symm_p = np.zeros(num_participants)
nll_array_rw_symm_p = np.zeros(num_participants)
aic_array_rw_symm_p = np.zeros(num_participants)
bic_array_rw_symm_p = np.zeros(num_participants)

# Participant index
participant_idx = 0

# Loop over participants
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):

    # Current participant data only
    df_p = df[df['pid'] == participant]

    # Arrays for Random Model
    nll_array_random = np.zeros(max_sessions)
    aic_array_random = np.zeros(max_sessions)
    bic_array_random = np.zeros(max_sessions)

    # Arrays for Bias Model
    nll_array_bias = np.zeros(max_sessions)
    aic_array_bias = np.zeros(max_sessions)
    bic_array_bias = np.zeros(max_sessions)

    # Arrays for Win-Stay-Lose-Shift Model
    nll_array_win_stay = np.zeros(max_sessions)
    aic_array_win_stay = np.zeros(max_sessions)
    bic_array_win_stay = np.zeros(max_sessions)

    # Arrays for RW Symmetric Model
    alpha_array_rw_symm = np.zeros(max_sessions)
    sigma_array_rw_symm = np.zeros(max_sessions)
    nll_array_rw_symm = np.zeros(max_sessions)
    aic_array_rw_symm = np.zeros(max_sessions)
    bic_array_rw_symm = np.zeros(max_sessions)

    # Session index
    session_idx = 0

    # Loop over sessions
    for session in df_p.session.unique():

        # Get current session data, one row per trial
        df_s = df_p[df_p.session == session]
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        # Only feedback trials
        df_s = df_s[df_s.condition != 'baseline']

        # Get variables
        confidence = df_s.confidence.values
        feedback = df_s.feedback.values
        n_trials = len(df_s)

        # Random confidence model
        # print('random model')
        nll_random_model = fit_random_model(prediction_range=100,
                                            n_trials=n_trials)

        # Get BIC and AIC
        k = 0
        random_model_aic = 2*k + 2*nll_random_model
        random_model_bic = k*np.log(n_trials) + 2*nll_random_model

        nll_array_random[session_idx] = nll_random_model
        aic_array_random[session_idx] = random_model_aic
        bic_array_random[session_idx] = random_model_bic

        # Biased confidence model
        # print('biased model...')
        # Set Bounds
        mean_bound = (1, 100)   # Mean
        sigma_bound = (1, 100)  # Standard deviation
        bounds = [(mean_bound[0], mean_bound[1]),
                  (sigma_bound[0],  sigma_bound[1]),
                  ]

        # Get results
        results = fit_model(model=random_model_w_bias,
                            args=(confidence,
                                  n_trials),
                            bounds=bounds,
                            n_trials=n_trials,
                            start_value_number=10,
                            solver="L-BFGS-B")

        best_mean = results[0]
        best_std = results[1]
        nll = results[2]
        aic = results[3]
        bic = results[4]
        pseudo_r2 = results[5]
        nll_array_bias[session_idx] = nll
        aic_array_bias[session_idx] = aic
        bic_array_bias[session_idx] = bic

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
                                     start_value_number=10,
                                     solver="L-BFGS-B")

        best_std = results_win_stay[0]
        best_win_boundary = results_win_stay[1]
        nll_array_win_stay[session_idx] = results_win_stay[2]
        aic_array_win_stay[session_idx] = results_win_stay[3]
        bic_array_win_stay[session_idx] = results_win_stay[4]

        # Rescorla wagner model

        # Set bounds
        alpha_bound = (0.001, 1)  # Alpha
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
                                    start_value_number=10,
                                    solver="L-BFGS-B")

        best_alpha = results_rw_symm[0]
        best_std = results_rw_symm[1]
        best_bias = results_rw_symm[2]
        nll = results_rw_symm[3]
        aic = results_rw_symm[4]
        bic = results_rw_symm[5]
        pseudo_r2 = results_rw_symm[6]

        alpha_array_rw_symm[session_idx] = best_alpha
        sigma_array_rw_symm[session_idx] = best_std
        nll_array_rw_symm[session_idx] = nll
        aic_array_rw_symm[session_idx] = aic
        bic_array_rw_symm[session_idx] = bic

        # Increment session index
        session_idx += 1

    # Compute and store participant-level average metrics
    nll_array_random_p[participant_idx] = np.mean(nll_array_random)
    aic_array_random_p[participant_idx] = np.mean(aic_array_random)
    bic_array_random_p[participant_idx] = np.mean(bic_array_random)

    nll_array_bias_p[participant_idx] = np.mean(nll_array_bias)
    aic_array_bias_p[participant_idx] = np.mean(aic_array_bias)
    bic_array_bias_p[participant_idx] = np.mean(bic_array_bias)

    nll_array_win_stay_p[participant_idx] = np.mean(nll_array_win_stay)
    aic_array_win_stay_p[participant_idx] = np.mean(aic_array_win_stay)
    bic_array_win_stay_p[participant_idx] = np.mean(bic_array_win_stay)

    alpha_array_rw_symm_p[participant_idx] = np.mean(alpha_array_rw_symm)
    sigma_array_rw_symm_p[participant_idx] = np.mean(sigma_array_rw_symm)
    nll_array_rw_symm_p[participant_idx] = np.mean(nll_array_rw_symm)
    aic_array_rw_symm_p[participant_idx] = np.mean(aic_array_rw_symm)
    bic_array_rw_symm_p[participant_idx] = np.mean(bic_array_rw_symm)

    # Increment participant index
    participant_idx += 1

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

# %% Plot mean across conditions

fig, ax = plt.subplots(figsize=(6, 4))

# Define x-coordinates and offsets for each model within the group
x = np.arange(3)  # Base x-coordinates for metrics
offset = 0.2  # Offset for each model within a group

# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple']

# Plotting NLL for each model
model_means = [
    random_model_mean_nll, bias_model_mean_nll,
    win_stay_model_mean_nll, rw_symm_model_mean_nll
]
model_sems = [
    random_model_sem_nll, bias_model_sem_nll,
    win_stay_model_sem_nll, rw_symm_model_sem_nll
]
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW Symmetric LR"
]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(
        x[0] + offset * (i - 1.5), mean, yerr=sem,
        fmt='o', capsize=10, color=colors[i], label=model_names[i]
    )

# Plotting AIC for each model
model_means = [
    random_model_mean_aic, bias_model_mean_aic,
    win_stay_model_mean_aic, rw_symm_model_mean_aic
]
model_sems = [
    random_model_sem_aic, bias_model_sem_aic,
    win_stay_model_sem_aic, rw_symm_model_sem_aic
]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(
        x[1] + offset * (i - 1.5), mean, yerr=sem,
        fmt='o', capsize=10, color=colors[i]
    )

# Plotting BIC for each model
model_means = [
    random_model_mean_bic, bias_model_mean_bic,
    win_stay_model_mean_bic, rw_symm_model_mean_bic
]
model_sems = [
    random_model_sem_bic, bias_model_sem_bic,
    win_stay_model_sem_bic, rw_symm_model_sem_bic
]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(
        x[2] + offset * (i - 1.5), mean, yerr=sem,
        fmt='o', capsize=10, color=colors[i]
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

plt.tight_layout()
plt.show()
