# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:33:30 2023

@author: carll
"""

# Analysis

import numpy as np
from src.utils import (add_session_column, load_df)
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import mannwhitneyu

# Import data
# =============================================================================
# current_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.dirname(current_directory)
# grandparent_directory = os.path.dirname(parent_directory)
# project_path = grandparent_directory
# experiment_path = r"fixed_feedback"
# fixed_feedback_data_path = r'data/cleaned'
# data_file = r'main-20-12-14-processed_filtered.csv'
# full_path = os.path.join(project_path, experiment_path,
#                          fixed_feedback_data_path, data_file)
# df = pd.read_csv(full_path, low_memory=False)
# =============================================================================

df = load_df(EXP=1)

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

# Save dataframe
# =============================================================================
# save_path = os.path.join(project_path, r"Fixed_feedback_data.csv")
# df_a.to_csv(save_path, index=False)
# =============================================================================
#save_path = os.path.join(project_path, 'Fixed_feedback_data.json')
#df_a.to_json(save_path, orient='records', lines=True)
# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

#%%

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}


# Plotting
fig, ax = plt.subplots(figsize=(5, 4))

# Set the font size globally
#plt.rcParams.update({'font.size': 15})
#plt.rcParams['xtick.labelsize'] = 15 # For x-axis tick labels
#lt.rcParams['ytick.labelsize'] = 15  # For y-axis tick labels

# Process trial and baseline data
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]
    num_participants = len(df_cond)

    if num_participants == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = [np.mean([participant[i]
                           for participant in df_cond['confidence_task']])
                  for i in range(20)]
    sem_trial = [np.std([participant[i]
                         for participant in df_cond['confidence_task']],
                        ddof=1) / np.sqrt(num_participants)
                 for i in range(20)]

    # Process baseline data: Use the last 10 trials for each participant
    last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(num_participants)

    # Adjust baseline and trial data for plotting
    x_baseline = list(range(-10, 0))
    x_trial = list(range(0, 20))
    x_combined = x_baseline + x_trial

    # Combine the baseline and trial data
    combined_mean = list(baseline_data) + trial_data
    combined_sem = list(baseline_sem) + sem_trial

    # Plot combined baseline and trial data as a continuous line
    ax.plot(x_combined, combined_mean, label=f'{cond.capitalize()}',
            color=colors[cond], lw=2, marker='o')
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

#ax.set_title('Mean Confidence Across Conditions')
ax.set_xlabel('Trials relative to start of feedback trials')
ax.set_ylabel('Confidence (mean+-SEM)')
ax.legend(fontsize=14)
ax.spines[['top', 'right']].set_visible(False)
ax.axvline(0, ls='--', c='k', label='Start of Task Trials',
           alpha=0.5)

plt.tight_layout()
save_folder = r"C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/analysis"
file_name = 'mean_confidence_over_trials'
save_path = os.path.join(save_folder, file_name)
fig.savefig(f'{save_path}.png', dpi=300)
plt.show()

#%%

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}

# Plotting
fig, ax = plt.subplots(figsize=(5, 4))

# Process trial and baseline data
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]
    num_participants = len(df_cond)

    if num_participants == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = [np.mean([participant[i]
                           for participant in df_cond['p_avg_task']])
                  for i in range(20)]
    sem_trial = [np.std([participant[i]
                         for participant in df_cond['p_avg_task']],
                        ddof=1) / np.sqrt(num_participants)
                 for i in range(20)]

    # Process baseline data: Use the last 10 trials for each participant
    last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(num_participants)

    # Adjust baseline and trial data for plotting
    x_baseline = list(range(-10, 0))
    x_trial = list(range(0, 20))
    x_combined = x_baseline + x_trial

    # Combine the baseline and trial data
    combined_mean = list(baseline_data) + trial_data
    combined_sem = list(baseline_sem) + sem_trial

    # Plot combined baseline and trial data as a continuous line
    ax.plot(x_combined, combined_mean, label=f'{cond.capitalize()}',
            color=colors[cond], lw=2, marker='o')
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

#ax.set_title('Performance')
ax.set_xlabel('Trials relative to start of feedback trials')
ax.set_ylabel('Performance (mean+-SEM)')
ax.legend(fontsize=14)
ax.spines[['top', 'right']].set_visible(False)
ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

plt.tight_layout()
save_folder = r"C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/analysis"
file_name = 'mean_performance_over_trials'
save_path = os.path.join(save_folder, file_name)
fig.savefig(f'{save_path}.png', dpi=300)
plt.show()
#%%

# Get BDI
df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

# Compute the median 'bdi' score
median_bdi = df_a['bdi'].median()

# Create a new column 'bdi_level' with 'high' or 'low' based on the median split
df_a['bdi_level'] = ['high' if x > median_bdi else 'low' for x in df_a['bdi']]


# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}

mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}
bdi_levels = ['high', 'low']

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Process trial and baseline data
for cond in conditions:
    for bdi_level in bdi_levels:
        # Filter data for current condition and bdi_level
        df_cond = df_a[(df_a['condition_list'] == cond) &
                       (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            print(f"No data available for condition {cond}")
            continue

        # Calculate mean and SEM for trial data
        trial_data = [np.mean([participant[i]
                               for participant in df_cond['confidence_task']])
                      for i in range(20)]
        sem_trial = [np.std([participant[i]
                             for participant in df_cond['confidence_task']],
                            ddof=1) / np.sqrt(num_participants)
                     for i in range(20)]

        # Process baseline data: Use the last 10 trials for each participant
        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        padded_baseline_array = np.array(last_10_baseline.tolist())

        # Calculate mean and SEM for baseline data
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array,
                                 axis=0, ddof=1) / np.sqrt(num_participants)

        # Adjust baseline and trial data for plotting
        x_baseline = list(range(-10, 0))
        x_trial = list(range(0, 20))
        x_combined = x_baseline + x_trial

        # Combine the baseline and trial data
        combined_mean = list(baseline_data) + trial_data
        combined_sem = list(baseline_sem) + sem_trial

        # Plot combined baseline and trial data as a continuous line
        ax.plot(x_combined, combined_mean,
                label=f'{cond.capitalize()}, BDI {bdi_level}',
                color=colors[f'{cond.capitalize()}, BDI {bdi_level}'],
                lw=2, marker='o')
        ax.fill_between(x_combined,
                        [m - s for m, s in zip(combined_mean, combined_sem)],
                        [m + s for m, s in zip(combined_mean, combined_sem)],
                        color=colors[f'{cond.capitalize()}, BDI {bdi_level}'],
                        alpha=0.2)

ax.set_title('Mean Confidence Across Conditions')
ax.set_xlabel('Trials relative to start of task trials')
ax.set_ylabel('Confidence (mean+-SEM)')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

# Assuming df_a is your DataFrame and it has 'condition_list', 'bdi_level', 'confidence_task', and 'confidence_baseline' columns

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {
    'Neut, BDI high': 'black',
    'Neut, BDI low': 'grey',
    'Pos, BDI high': 'darkgreen',
    'Pos, BDI low': 'lime',
    'Neg, BDI high': 'darkred',
    'Neg, BDI low': 'tomato'
}
bdi_levels = ['high', 'low']

# Plotting - create a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition
    for bdi_level in bdi_levels:
        # Filter data for current condition and bdi_level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial data
        trial_data = [np.mean([participant[i] for participant in df_cond['confidence_task']])
                      for i in range(20)]
        sem_trial = [np.std([participant[i] for participant in df_cond['confidence_task']], ddof=1) / np.sqrt(num_participants)
                     for i in range(20)]

        # Adjust baseline and trial data for plotting
        x_trial = list(range(0, 20))

        # Plot trial data as a continuous line
        ax.plot(x_trial, trial_data, label=f'BDI {bdi_level}', color=colors[f'{cond.capitalize()}, BDI {bdi_level}'], lw=2, marker='o')
        ax.fill_between(x_trial, [m - s for m, s in zip(trial_data, sem_trial)],
                        [m + s for m, s in zip(trial_data, sem_trial)],
                        color=colors[f'{cond.capitalize()}, BDI {bdi_level}'], alpha=0.2)

    ax.set_title(f'{cond.capitalize()} Condition')
    ax.set_xlabel('Trial Number')
    if i == 0:  # Add y-axis label only to the first subplot to avoid repetition
        ax.set_ylabel('Confidence (mean±SEM)')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

plt.tight_layout()
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt

# Assuming df_a is your DataFrame and it has 'condition_list', 'bdi_level', 'confidence_task', and 'confidence_baseline' columns

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([participant for participant in df_cond['confidence_task']], axis=0)
        sem_trial = np.std([participant for participant in df_cond['confidence_task']], axis=0, ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        padded_baseline_array = np.array(last_10_baseline.tolist())
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        legend_label = f'BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=legend_label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    if cond == 'neut':
        title = 'Neutral'
    if cond == 'neg':
        title = 'Negative'
    if cond == 'pos':
        title = 'Positive'

    ax.set_title(f'{title} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Confidence (mean±SEM)')
    ax.set_xlabel('Trials')

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% Alignt traces

import numpy as np
import matplotlib.pyplot as plt

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([participant for participant in df_cond['confidence_task']], axis=0)
        sem_trial = np.std([participant for participant in df_cond['confidence_task']], axis=0, ddof=1) / np.sqrt(num_participants)

        # Normalize trial data around the mean of the first task trial
        norm_factor = trial_data[0]  # Mean of the first task trial
        trial_data -= norm_factor  # Normalize trial data

        # Normalize baseline data similarly
        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        padded_baseline_array = np.array(last_10_baseline.tolist())
        baseline_data = np.nanmean(padded_baseline_array, axis=0) - norm_factor  # Normalize baseline data
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine normalized baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Change in Confidence (mean±SEM)')
    ax.set_xlabel('Trials')

# Remove top and right spines and add legend
for ax in axs:
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()

plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt


# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([participant for participant in df_cond['p_avg_task']], axis=0)
        sem_trial = np.std([participant for participant in df_cond['p_avg_task']], axis=0, ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
        padded_baseline_array = np.array(last_10_baseline.tolist())
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Relative Performance (mean±SEM)')
    ax.set_xlabel('Trials')



# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% BDI difference in relative performance over time

import numpy as np
import matplotlib.pyplot as plt

# Assuming df_a is your DataFrame and it has 'condition_list', 'bdi_level', 'confidence_task', and 'confidence_baseline' columns

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([participant for participant in df_cond['p_avg_task']], axis=0)
        sem_trial = np.std([participant for participant in df_cond['p_avg_task']], axis=0, ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
        padded_baseline_array = np.array(last_10_baseline.tolist())
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        legend_label = f'BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=legend_label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    if cond == 'neut':
        title = 'Neutral'
    if cond == 'neg':
        title = 'Negative'
    if cond == 'pos':
        title = 'Positive'

    ax.set_title(f'{title} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Relative Performance (mean±SEM)')
    ax.set_xlabel('Trials')

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% BDI difference in absolute performance over time

import numpy as np
import matplotlib.pyplot as plt

# Assuming df_a is your DataFrame and it has 'condition_list', 'bdi_level', 'confidence_task', and 'confidence_baseline' columns

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([participant for participant in df_cond['performance_task']], axis=0)
        sem_trial = np.std([participant for participant in df_cond['performance_task']], axis=0, ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['performance_baseline'].apply(lambda x: x[-10:])
        padded_baseline_array = np.array(last_10_baseline.tolist())
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        legend_label = f'BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=legend_label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    if cond == 'neut':
        title = 'Neutral'
    if cond == 'neg':
        title = 'Negative'
    if cond == 'pos':
        title = 'Positive'

    ax.set_title(f'{title} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Absolute Performance (mean±SEM)')
    ax.set_xlabel('Trials')

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()


#%% learning to align confidence with relative performance across time

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([(np.mean(p_perf)-p_conf)
                              for p_conf, p_perf
                              in zip(df_cond['confidence_task'],
                                     df_cond['p_avg_task'])], axis=0)
        sem_trial = np.std([(np.mean(p_perf)-p_conf)
                              for p_conf, p_perf
                              in zip(df_cond['confidence_task'],
                                     df_cond['p_avg_task'])], axis=0,
                           ddof=1) / np.sqrt(num_participants)

        conf_last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        p_avg_last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diff = [(np.mean(p_perf)-p_conf)
                              for p_conf, p_perf
                              in zip(conf_last_10_baseline,
                                     p_avg_last_10_baseline)]
        padded_baseline_array = np.array(last_10_baseline_diff)
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('performance - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% learning to align confidence with absolute error across time

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) &
                       (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        df_cond['performance_task'] = df_cond['performance_task']*-1
        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([abs(np.mean(p_perf)-p_conf)
                              for p_conf, p_perf
                              in zip(df_cond['confidence_task'],
                                     df_cond['performance_task'])], axis=0)
        sem_trial = np.std([abs(np.mean(p_perf)-p_conf)
                              for p_conf, p_perf
                              in zip(df_cond['confidence_task'],
                                     df_cond['performance_task'])], axis=0,
                           ddof=1) / np.sqrt(num_participants)


        conf_last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        p_avg_last_10_baseline = df_cond['performance_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diff = [abs(p_perf-p_conf)
                              for p_conf, p_perf
                              in zip(conf_last_10_baseline,
                                     p_avg_last_10_baseline)]
        padded_baseline_array = np.array(last_10_baseline_diff)
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('performance - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% learning to align confidence with feedback across time

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([abs(np.mean(p_fb)-p_conf[:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['feedback'])],
                             axis=0)
        sem_trial = np.std([abs(np.mean(p_fb)-p_conf[:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['feedback'])],
                           axis=0,
                           ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diffs = [abs(np.mean(0)-p_conf[:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_baseline'],
                                     df_cond['feedback'])]

        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Feedback - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()



#%% learning to align confidence with feedback across time
# Relative difference

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([(np.mean(p_fb)-p_conf[:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['feedback'])],
                             axis=0)
        sem_trial = np.std([(np.mean(p_fb)-p_conf[:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['feedback'])],
                           axis=0,
                           ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diffs = [(0-p_conf)
                              for p_conf
                              in df_cond['confidence_baseline']]
        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Feedback - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% learning to align confidence with feedback across time
# Plot each trace!!!


# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = [abs(np.mean(p_fb)-p_conf[1:])
                      for p_conf, p_fb
                      in zip(df_cond['confidence_task'],
                             df_cond['feedback'])]

# =============================================================================
#         sem_trial = np.std([abs(p_fb[:-1]-p_conf[1:])
#                               for p_conf, p_fb
#                               in zip(df_cond['confidence_task'],
#                                      df_cond['feedback'])],
#                            axis=0,
#                            ddof=1) / np.sqrt(num_participants)
# =============================================================================

        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diffs = [abs(0-p_conf)
                                  for p_conf
                                  in df_cond['confidence_baseline']]

        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_data = padded_baseline_array


        # Combine baseline and trial data for plotting
        combined_mean = [np.concatenate((b, t)) for b, t in zip(baseline_data,
                                                                trial_data)]


        # Test error vs conf
        trial_error_data = [abs(np.mean(p_fb)-p_conf[1:])
                           for p_conf, p_fb
                           in zip(df_cond['confidence_task'],
                                 df_cond['performance_task'])]

        last_10_baseline = df_cond['performance_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diffs = [abs(0-p_conf)
                                  for p_conf
                                  in df_cond['performance_baseline']]

        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_error_data = padded_baseline_array

        # Combine baseline and trial data for plotting
        combined_error_mean = [np.concatenate((b, t))
                               for b, t
                               in zip(baseline_error_data,
                                      trial_error_data)]

        #
        combined_mean = np.array(combined_mean) #np.array(combined_error_mean)

        #combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data[0])))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        for i in range(int((len(combined_mean)/100)*4)):

            ax.plot(x_combined, combined_mean[i], label=label,
                    color=colors[label],
                    lw=1, marker='')
        #ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Feedback - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    #axs[i].legend()

plt.tight_layout()
plt.show()


#%% learning to align confidence with feedback across time
# Plot each trace!!!

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import pearsonr
import statsmodels.api as sm

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut': 'black',
          'Neut': 'grey',
          'Pos': 'darkgreen',
          'Pos': 'lime',
          'Neg': 'darkred',
          'Neg': 'tomato'}

conditions = ['neut', 'pos', 'neg']
cmap = plt.cm.Reds  # Choose a colormap
norm = Normalize(vmin=df_a['bdi'].min(), vmax=20#df_a['bdi'].max()
                 )  # Normalize BDI scores
scalarmap = ScalarMappable(norm=norm, cmap=cmap)

# Creating a subplot for each condition
fig, axs = plt.subplots(2, len(conditions), figsize=(15, 12), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[0,i]  # Select the subplot for the current condition
    ax2 = axs[1,i]

    df_cond = df_a[(df_a['condition_list'] == cond)]
    num_participants = len(df_cond['pid'].unique())
    trial_means = []
    baseline_means = []
    BDIs= []
    metacogbias = []
    for pid in df_cond.pid.unique():
        # Filter data for current condition and BDI level
        df_p = df_cond[df_cond['pid'] == pid]

        # Get trial data in list
        trial_data = [(np.mean(p_fb)-p_conf[1:])
                      for p_conf, p_fb
                      in zip(df_p['confidence_task'],
                             df_p['feedback'])]

        trial_mean = np.mean(trial_data[0])
        trial_means.append(trial_mean)

        last_10_baseline = df_p['confidence_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diffs = [(0-p_conf)
                                  for p_conf
                                  in df_p['confidence_baseline']]

        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_data = padded_baseline_array
        baseline_means.append(np.mean(baseline_data))

        # Combine baseline and trial data for plotting
        combined_mean = [np.concatenate((b, t)) for b, t in zip(baseline_data,
                                                                trial_data)]

        #combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data[0])))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}'

        # set color to bdi
        color = scalarmap.to_rgba(df_p['bdi'].iloc[0])
        BDIs.append(df_p['bdi'].iloc[0])

        ax.plot(x_combined, combined_mean[0], label=label,
                color=color,
                lw=.5, marker='', alpha=0.7)


        # Estimate metacog bias
        df_model = pd.DataFrame(
            {'confidence_baseline': df_p.confidence_baseline.values,
             'confidence_task': df_p.confidence_task.values,
             'performance_baseline': df_p.performance_baseline.values,
             'performance_task': df_p.performance_task.values})

        X = sm.add_constant(df_model['performance_baseline'][0])

        # Define the dependent variable
        y = df_model['confidence_baseline'][0]

        # Fit the linear regression model
        model = sm.OLS(y, X).fit()

        # Save the intercept (metacognitive bias)
        metacogbias.append(model.params[0])

        # Print the summary of the model
       # print(model.summary())


    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Feedback - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

    y = metacogbias

    scatter = ax2.scatter(BDIs, y)
    ax2.set_xlabel('BDI')
    ax2.set_ylabel('Mean alignment')

    # Perform Pearson correlation
    corr, p_value = pearsonr(BDIs, y)

    # Annotate the plot with the correlation coefficient and p-value
    ax2.annotate(f'Pearson r={corr:.2f}\np-value={p_value:.3f}',
                 xy=(0.1, -0.7), xycoords='axes fraction',
                 ha='left', va='top')

# Remove top right spines and add legend
for i in range(len(axs)+1):
    axs[0, i].spines[['top', 'right']].set_visible(False)
    axs[1, i].spines[['top', 'right']].set_visible(False)
   # axs[i].legend()

plt.tight_layout()
plt.show()


#%% cumulative average vector

def cumulative_average(vector):
    cumsum = 0  # To hold the cumulative sum
    cumavg = []  # To hold the cumulative average values
    for i, value in enumerate(vector, start=1):
        cumsum += value
        cumavg.append(cumsum / i)  # Append the cumulative average to the list
    return cumavg

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(15, 5), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) & (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())
        # Apply the cumulative_average function to each row in the 'feedback' column
        df_cond.loc[:, 'cum_avg_feedback'] = df_cond['feedback'].apply(cumulative_average)

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([(p_fb[:-1]-p_conf[1:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['cum_avg_feedback'])], axis=0)
        sem_trial = np.std([(p_fb[:-1]-p_conf[1:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['cum_avg_feedback'])], axis=0,
                           ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
        last_10_baseline_diffs = [(0-p_conf)
                              for p_conf
                              in df_cond['confidence_baseline']]
        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array, axis=0,
                                 ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label, color=colors[label],
                lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem,
                        combined_mean + combined_sem, color=colors[label],
                        alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')  # Mark start of task trials
    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Feedback - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% Mixed Linear Model Predicting Confidence
#   from previous feedback and curren performance

import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd

def cumulative_average(vector):
    cumsum = 0  # To hold the cumulative sum
    cumavg = []  # To hold the cumulative average values
    for i, value in enumerate(vector, start=1):
        cumsum += value
        cumavg.append(cumsum / i)  # Append the cumulative average to the list
    return cumavg


# Restructure df
df = df_a.copy()

# Get bdi score for each participant
bdi_scores = df_a.groupby('pid')['bdi'].first().reset_index()

# Create trial data where only single numbers
df['trial'] = [np.array(range(20)) for i in range(len(df))]
df['session'] = [[i]*20 for i in df['session'].values]
df['pid'] = [[i]*20 for i in df['pid'].values]
df['condition'] = [[i]*20 for i in df['condition_list'].values]

# Create new column
df.loc[:, 'cum_avg_feedback'] = df['feedback'].apply(cumulative_average)

# Create dict
data = {
    'confidence_task': df['confidence_task'].values,
    'feedback': df['feedback'].values,
    'p_avg_task': df['p_avg_task'].values,
    'performance_task': df['performance_task'].values,
    'trial': df['trial'].values,
    'pid': df['pid'].values,  # PIDs
    'session': df['session'].values,
    'condition': df['condition'].values,
}

# Create a DataFrame
df_cond_short = pd.DataFrame(data)

# Initialize an empty DataFrame for the expanded data
expanded_df = pd.DataFrame()

# Iterate over each row and expand it into multiple rows
for index, row in df_cond_short.iterrows():
    # Create a DataFrame from lists
    temp_df = pd.DataFrame({
        'confidence_task': row['confidence_task'],
        'feedback': row['feedback'],
        'p_avg_task': row['p_avg_task'],
        'error': row['performance_task'],
        'trial': row['trial'],
        'pid': row['pid'],
        'session': row['session'],
        'condition': row['condition'],
        #'feedback_sub_confidence': row['feedback_sub_confidence'],
    })

    # Append to the expanded DataFrame
    expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)

# Add bdi score of each participant
#df_merged = pd.merge(df, bdi_scores, on='pid', how='left')

#%%

import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Specify dataframe
df = expanded_df.copy()

# Group by 'pid' and 'session', then shift the 'feedback' and 'performance'
# shift down by one to get 'previous_feedback' and 'previous_performance'
df['previous_feedback'] = df.groupby(['pid',
                                      'session']
                                     )['feedback'].shift(1)
df['previous_error'] = df.groupby(['pid',
                                   'session']
                                  )['error'].shift(1)

df['previous_confidence'] = df.groupby(['pid',
                                   'session']
                                  )['confidence_task'].shift(1)

df['delta_rule'] = (df['previous_feedback'] - df['previous_confidence']) + df['previous_confidence']

# Create a mask that identifies the first row in each group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0

# Use the mask to filter out the first row of each group
df = df[~is_first_row].reset_index(drop=True)

# Unique conditions in the DataFrame
conditions = df['condition'].unique()

# Dictionary to store the fitted models for later reference
fitted_models = {}

# Fit a mixed model for each condition
for condition in conditions:

    # Filter the DataFrame for the current condition
    df_cond = df[df['condition'] == condition]

    # Define the model formula
    model_formula = 'confidence_task ~ delta_rule + error + previous_error'

    # Fit the mixed model with 'pid' as a random effect
    mixed_model = smf.mixedlm(model_formula, df_cond,
                              groups=df_cond['pid']).fit()

    # Store the fitted model
    fitted_models[condition] = mixed_model

    # Print summary for inspection (optional)
    print(f"Condition: {condition}")
    print(mixed_model.summary())
    print("\n" + "="*80 + "\n")

# Predictor variables of interest
predictors = ['delta_rule', 'error', 'previous_error']

# Define the colors for each condition
condition_colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Setup subplots - one subplot per predictor
fig, axs = plt.subplots(len(predictors), 1, figsize=(4, 2 * len(predictors)),
                        sharex=True)

# Loop over each predictor to create a subplot
for i, predictor in enumerate(predictors):
    ax = axs[i] if len(predictors) > 1 else axs
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(0,100)
    # Loop over each condition to plot in the same subplot
    for j, condition in enumerate(conditions):
        model = fitted_models[condition]

        # Create x values range based on the predictor
        #x_values = np.linspace(df[predictor].min(), df[predictor].max(), 100)
        x_values = np.linspace(0, 100, 100)

        # Get coef values
        intercept = model.params.iloc[0]
        slope = model.params.loc[predictor]

        # Calculate y values for the trend line
        y_values = intercept + slope * x_values

        # Plot the trend line for the condition in the current subplot
        ax.plot(x_values, y_values, label=f'{condition}',
                color=condition_colors[condition])

        # Check significance and add an asterisk if significant
        if model.pvalues.loc[predictor] < 0.05:
            mid_point = len(x_values) // 2
            ax.text(x_values[mid_point]+j, y_values[mid_point],
                    '*', fontsize=15, color=condition_colors[condition],
                    verticalalignment='bottom')

    # Set titles and labels for the subplot
    ax.set_xlabel(predictor)
    ax.set_ylabel('Confidence')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

# Adjust layout
plt.tight_layout()

# Save plot
save_folder = r'comparative_models\results\Fixed_feedback\Regressions'
file_name = r'LMM_conf_~_feedback_and_performance.png'
save_path = os.path.join(project_path,save_folder,file_name)
plt.savefig(save_path, dpi=300)

# Show plot
plt.show()

#%% Correlation between predictors?
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Select only the columns that are predictors
predictors_df = df[['delta_rule', 'error', 'previous_error',]]

# Initialize empty matrices for correlations and p-values
corr_matrix = predictors_df.corr()
p_value_matrix = pd.DataFrame(data=np.zeros(shape=(len(predictors_df.columns),
                                                   len(predictors_df.columns))),
                              columns=predictors_df.columns,
                              index=predictors_df.columns)

# Populate the p-value matrix
for row in predictors_df.columns:
    for col in predictors_df.columns:
        if row != col:
            _, p_value = pearsonr(predictors_df[row], predictors_df[col])
            p_value_matrix.loc[row, col] = p_value
        else:
            # Set the diagonal to NaNs
            p_value_matrix.loc[row, col] = np.nan

# Define a mask for significant p-values
mask_significant = p_value_matrix < 0.05

# Plotting the heatmap with a mask for non-significant correlations
plt.figure(figsize=(9, 9))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
            mask=~mask_significant, cbar_kws={'label': 'Pearson Correlation'},
            vmin=-1, vmax=1)

# Optional: Add asterisks or other markers to denote significance
for i, row in enumerate(predictors_df.columns):
    for j, col in enumerate(predictors_df.columns):
        if mask_significant.loc[row, col]:
            plt.text(j+0.5, i+0.4,
                     '*',
                     ha='center',
                     va='center',
                     color='white')
# =============================================================================
#
# plt.title('Correlation Matrix of Predictors')
# plt.show()
# =============================================================================


# Assuming 'df' is your DataFrame and includes the predictors
predictors_df = df[['delta_rule', 'error', 'previous_error',]]

# Compute the correlation matrix for the heatmap
corr_matrix = predictors_df.corr()

# Calculate VIF for the predictors
predictors_df = add_constant(predictors_df)  # Add constant for VIF calculation
vif_data = pd.DataFrame({
    'Predictor': predictors_df.columns,
    'VIF': [round(variance_inflation_factor(predictors_df.values, i), 4)
            for i in range(predictors_df.shape[1])]
})

# Adding the VIF table below the heatmap
# Remove the constant row from the VIF data (first row)
vif_data = vif_data.iloc[1:]
# Create the table
plt.table(cellText=vif_data.values,
          colLabels=vif_data.columns,
          cellLoc='center',
          loc='bottom',
          bbox=[-0, -0.6, 1, 0.3])  # bbox=[left, bottom, width, height]

# Adjust layout to make room for the table
#plt.subplots_adjust(bottom=0.0)
plt.tight_layout()

# Save plot
save_folder = r'comparative_models\results\Fixed_feedback\Regressions'
file_name = r'Corr_matrix and VIF LMM_conf_~_feedback_and_performance.png'
save_path = os.path.join(project_path,save_folder,file_name)
plt.savefig(save_path, dpi=300)

plt.show()

#%% Confidence alignment across time

df = expanded_df.copy()

# Group by 'pid' and 'session', then shift the 'feedback' and 'performance'
# columns down by one to get 'previous_feedback' and 'previous_performance'
df['previous_feedback'] = df.groupby(['pid',
                                      'session']
                                     )['feedback'].shift(1)
df['previous_error'] = df.groupby(['pid',
                                   'session']
                                  )['error'].shift(1)

# Create a mask that identifies the first row in each group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0

# Use the mask to filter out the first row of each group
df = df[~is_first_row].reset_index(drop=True)

# Add column showing aligmnet between feedback and confidence
df['feedback_minus_confidence'] = abs(df['previous_feedback'] - df['confidence_task'])

df['error_minus_confidence'] = abs(df['error']*-1 - df['confidence_task'])

df['rel_performance_minus_confidence'] = abs(df['p_avg_task'] - df['confidence_task'])


# Unique conditions in the DataFrame
conditions = df['condition'].unique()

# Dictionary to store the fitted models for later reference
fitted_models = {}

# Fit a mixed model for each condition
for condition in conditions:

    # Filter the DataFrame for the current condition
    df_cond = df[df['condition'] == condition]

    # Define the model formula
    model_formula = 'error_minus_confidence ~ trial'

    # Fit the mixed model with 'pid' as a random effect
    mixed_model = smf.mixedlm(model_formula, df_cond,
                              groups=df_cond['pid']).fit()

    # Store the fitted model
    fitted_models[condition] = mixed_model

    # Print summary for inspection (optional)
    print(f"Condition: {condition}")
    print(mixed_model.summary())
    print("\n" + "="*80 + "\n")

# Predictor variables of interest
predictors = ['trial']

# Define the colors for each condition
condition_colors = {'neut': 'grey',
                    'pos': 'green',
                    'neg': 'red'}

# Setup subplots - one subplot per predictor
fig, axs = plt.subplots(len(predictors), 1, figsize=(4, 2 * len(predictors)),
                        sharex=True)

# Loop over each predictor to create a subplot
for i, predictor in enumerate(predictors):
    ax = axs[i] if len(predictors) > 1 else axs
    ax.spines[['top', 'right']].set_visible(False)
    #ax.set_xlim(0,100)
    # Loop over each condition to plot in the same subplot
    for j, condition in enumerate(conditions):
        model = fitted_models[condition]

        # Create x values range based on the predictor
        x_values = np.linspace(df[predictor].min(), df[predictor].max(), 100)
       # x_values = np.linspace(, 100, 100)

        # Get coef values
        intercept = model.params.iloc[0]
        slope = model.params.loc[predictor]

        # Calculate y values for the trend line
        y_values = intercept + slope * x_values

        # Plot the trend line for the condition in the current subplot
        ax.plot(x_values, y_values, label=f'{condition}',
                color=condition_colors[condition])

        # Check significance and add an asterisk if significant
        if model.pvalues.loc[predictor] < 0.05:
            mid_point = len(x_values) // 2
            ax.text(x_values[mid_point]+j, y_values[mid_point],
                    '*', fontsize=15, color=condition_colors[condition],
                    verticalalignment='bottom')

    # Set titles and labels for the subplot
    ax.set_xlabel(predictor)
    ax.set_ylabel('Error - Confidence')
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))

# Adjust layout
#plt.tight_layout()

# Save plot
save_folder = r'comparative_models\results\Fixed_feedback\Regressions'
file_name = r'LMM_feedback_minus_conf_~_trial.png'
save_path = os.path.join(project_path,save_folder,file_name)
plt.savefig(save_path, dpi=300)

# Show plot
plt.show()


#%%
# =============================================================================
#
# # Initialize variables
# conditions = ['neut', 'pos', 'neg']
# colors = {'Neut, BDI high': 'black',
#           'Neut, BDI low': 'grey',
#           'Pos, BDI high': 'darkgreen',
#           'Pos, BDI low': 'lime',
#           'Neg, BDI high': 'darkred',
#           'Neg, BDI low': 'tomato'}
#
# bdi_levels = ['high', 'low']
#
# # Specify dataframe
# df = expanded_df.copy()
#
# # Group by 'pid' and 'session', then shift the 'feedback' and 'performance'
# # columns down by one to get 'previous_feedback' and 'previous_performance'
# df['previous_feedback'] = df.groupby(['pid',
#                                       'session']
#                                      )['feedback'].shift(1)
#
# # OBS! error here is the same as perforamnce_task  in df_a
# df['previous_performance'] = df.groupby(['pid',
#                                          'session']
#                                         )['error'].shift(1)
#
# # Initialize a plot
# plt.figure(figsize=(10, 6))
#
# for i, cond in enumerate(conditions):
#
#     df_cond = df[(df['condition'] == cond)]
#     num_participants = len(df_cond['pid'].unique())
#
#     # Sample data structure similar to yours
#     data = {
#         'confidence_task': df_cond['confidence_task'].values,  # example data
#         'previous_feedback': df_cond['previous_feedback'].values,  # example data
#         'p_avg_task': df_cond['p_avg_task'].values,  # example data
#         'performance_task': df_cond['performance_task'].values,  # example data
#         'previous_performance': df_cond['previous_performance'].values,  # example data
#         'pid': df_cond['pid'].values  # PIDs
#     }
#
#     # Create a DataFrame
#     df_cond_short = pd.DataFrame(data)
#
#     # Initialize an empty DataFrame for the expanded data
#     expanded_df = pd.DataFrame()
#
#     # Iterate over each row and expand it into multiple rows
#     for index, row in df_cond_short.iterrows():
#         # Create a DataFrame from lists
#         temp_df = pd.DataFrame({
#             'confidence_task': row['confidence_task'],
#             'previous_feedback': row['previous_feedback'],
#             'p_avg_task': row['p_avg_task'],
#             'performance_task': row['performance_task'],
#             'previous_performance': row['previous_performance'],
#             'pid': [row['pid']] * len(row['confidence_task'])
#         })
#
#         # Append to the expanded DataFrame
#         expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)
#
#     expanded_df['pid2'], _ = pd.factorize(expanded_df['pid'])
#
#     # Define the model formula
#     model_formula = 'confidence_task ~ previous_feedback + performance_task + previous_performance'
#
#     # Fit the mixed model
#     mixed_model = smf.mixedlm(model_formula, expanded_df, groups=expanded_df['pid']).fit()
#
#     # Print the model summary to inspect the results
#     print(mixed_model.summary())
#
#     is_significant = mixed_model.pvalues[1] < 0.05
#
#     # Plotting
#     # Generate a range of values for 'previous_feedback' to create a continuous line for the fit
#     x_values = np.linspace(expanded_df['previous_feedback'].min(), expanded_df['previous_feedback'].max(), 100)
#
#     # Calculate the predicted 'confidence_task' values based on the model's coefficients
#     # Intercept is at index 0, 'previous_feedback' coefficient is at index 1
#     y_values = mixed_model.params[0] + mixed_model.params[1] * x_values
#
#     # Plot the line
#     plt.plot(x_values, y_values, label=f'{cond.capitalize()} Condition', color=colors[f'{cond.capitalize()}, BDI high'])  # Using the 'high' color for simplicity
#
#     # If the effect is significant, add an asterisk at the midpoint of the line
#     if is_significant:
#         mid_point = len(x_values) // 2
#         plt.text(x_values[mid_point], y_values[mid_point], '*', fontsize=15, color='red')
#
# # Finalizing the plot
# plt.xlabel('Previous Feedback')
# plt.ylabel('Confidence Task')
# plt.title('Fit Lines per Condition with Significance Indication')
# plt.legend()
# plt.show()
#
# =============================================================================

#%%
# =============================================================================
#
# import pandas as pd
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
#
# df = df_a.copy()
#
# # Shift the 'feedback' column by 1 to get the previous trial's feedback
# # Group by 'pid' or session identifier to avoid data leakage across sessions
# df['previous_feedback'] = df.groupby(['pid'])['feedback'].shift(1)
#
# # Drop the first trial of each session/participant since it won't have previous trial feedback
# df = df.dropna(subset=['previous_feedback']).reset_index(drop=True)
#
# # Assuming 'condition_list' identifies the conditions and 'pid' is the participant identifier
# conditions = ['neut', 'pos', 'neg']
#
# for cond in conditions:
#     df_cond = df[df['condition_list'] == cond]
#
#     # Expanded DataFrame list
#     expanded_rows = []
#
#     for _, row in df_cond.iterrows():
#         for conf_task, prev_fb in zip(row['confidence_task'], row['previous_feedback']):
#             # Create a new row for each element in the list
#             expanded_rows.append({'confidence_task': conf_task, 'previous_feedback': prev_fb, 'pid': row['pid']})
#
#     # Convert the expanded_rows list to a DataFrame
#     expanded_df = pd.DataFrame(expanded_rows)
#
#     # Define the model formula
#     model_formula = 'confidence_task ~ previous_feedback + (1 | pid)'
#
#     # Fit the mixed model
#     mixed_model = smf.mixedlm(model_formula, expanded_df, groups=expanded_df['pid']).fit()
#
#     # Print the model summary to inspect the results
#     print(f"Condition: {cond}")
#     print(mixed_model.summary())
#     print("\n" + "="*80 + "\n")
#
#
# =============================================================================
#%% learning to align current confidence with current feedback across time - relative

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'Neut, BDI high': 'black',
          'Neut, BDI low': 'grey',
          'Pos, BDI high': 'darkgreen',
          'Pos, BDI low': 'lime',
          'Neg, BDI high': 'darkred',
          'Neg, BDI low': 'tomato'}
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, axs = plt.subplots(1, len(conditions), figsize=(13, 4), sharey=True)

# Iterate over each condition to create a subplot
for i, cond in enumerate(conditions):
    ax = axs[i]  # Select the subplot for the current condition

    for bdi_level in bdi_levels:
        # Filter data for current condition and BDI level
        df_cond = df_a[(df_a['condition_list'] == cond) &
                       (df_a['bdi_level'] == bdi_level)]
        num_participants = len(df_cond['pid'].unique())

        if num_participants == 0:
            continue  # Skip if no data available

        # Calculate mean and SEM for trial and baseline data
        trial_data = np.mean([(p_fb[:-1]-p_conf[1:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['feedback'])], axis=0)
        sem_trial = np.std([(p_fb[:-1]-p_conf[1:])
                              for p_conf, p_fb
                              in zip(df_cond['confidence_task'],
                                     df_cond['feedback'])], axis=0,
                           ddof=1) / np.sqrt(num_participants)

        last_10_baseline = df_cond['confidence_baseline'].apply(lambda x:
                                                                x[-10:])
        last_10_baseline_diffs = [(0-p_conf)
                              for p_conf
                              in df_cond['confidence_baseline']]
        padded_baseline_array = np.array(last_10_baseline_diffs)
        baseline_data = np.nanmean(padded_baseline_array, axis=0)
        baseline_sem = np.nanstd(padded_baseline_array,
                                 axis=0, ddof=1) / np.sqrt(num_participants)

        # Combine baseline and trial data for plotting
        combined_mean = np.concatenate((baseline_data, trial_data))
        combined_sem = np.concatenate((baseline_sem, sem_trial))
        x_combined = list(range(-10, len(trial_data)))

        # Plot combined baseline and trial data
        label = f'{cond.capitalize()}, BDI {bdi_level}'
        ax.plot(x_combined, combined_mean, label=label,
                color=colors[label], lw=2, marker='o')
        ax.fill_between(x_combined, combined_mean - combined_sem,
                        combined_mean + combined_sem, color=colors[label],
                        alpha=0.2)

    # Set subplot title and labels
    ax.set_title(f'{cond.capitalize()} Condition')
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

    if i == 0:  # Add y-label only to the first subplot for clarity
        ax.set_ylabel('Feedback - confidence\n(mean±SEM)')
    ax.set_xlabel('Trials')
    ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
for i in range(len(axs)):
    axs[i].spines[['top', 'right']].set_visible(False)
    axs[i].legend()

plt.tight_layout()
plt.show()

#%% no groups learning to align current confidence with current feedback across time

# Initialize variables
conditions = ['neut', 'pos', 'neg']
full_conditions = ['Neutral', 'Positive', 'Negative']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}

# Plotting
fig, [ax, ax2] = plt.subplots(1,2, figsize=(10, 4))

for cond, full_cond in zip(conditions, full_conditions):
    df_cond = df_a[df_a['condition_list'] == cond]
    num_participants = len(df_cond)

    if num_participants == 0:
        print(f"No data available for condition {cond}")
        continue

    # Get trial array
    trial_data_array = np.array([(np.mean(p_fb[:])-p_conf[:])
                                 for p_fb, p_conf
                                 in zip(df_cond['feedback'],
                                        df_cond['confidence_task'])])

    # Calculate mean and SEM for trial data
    trial_data = np.mean(trial_data_array, axis=0)
    sem_trial = np.std(trial_data_array,
                       axis=0, ddof=1) / np.sqrt(num_participants)

    # Process baseline data: Use the last 10 trials for each participant
    p_conf_last_10_baseline = df_cond['confidence_baseline'].apply(lambda x:
                                                                   x[-10:])
    baseline_array = np.array([(0-p_conf)
                                 for p_conf
                                 in p_conf_last_10_baseline])
    padded_baseline_array = baseline_array # np.array(baseline_array.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array, axis=0,
                             ddof=1) / np.sqrt(num_participants)

    # Combine the baseline and trial data
    combined_mean = np.concatenate([baseline_data, trial_data])
    combined_sem = np.concatenate([baseline_sem, sem_trial])
    x_combined = list(range(-10, len(combined_mean)-10))

    # Plot combined baseline and trial data as a continuous line
    ax.plot(x_combined, combined_mean, label=f'{full_cond.capitalize()}',
            color=colors[cond], lw=1, markersize=4, marker='o')
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

    # Get trial array now with absolute values
    trial_data_array = np.array([abs(np.mean(p_fb)-p_conf)
                                 for p_fb, p_conf
                                 in zip(df_cond['feedback'],
                                        df_cond['confidence_task'])])

    # Calculate mean and SEM for trial data
    trial_data = np.mean(trial_data_array, axis=0)
    sem_trial = np.std(trial_data_array,
                       axis=0, ddof=1) / np.sqrt(num_participants)

    # Process baseline data: Use the last 10 trials for each participant
    p_conf_last_10_baseline = df_cond['confidence_baseline'].apply(lambda
                                                                   x: x[-10:])
    baseline_array = np.array([abs(0-p_conf)
                                 for p_conf
                                 in p_conf_last_10_baseline])
    padded_baseline_array = baseline_array # np.array(baseline_array.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array, axis=0, ddof=1) / np.sqrt(num_participants)

    # Combine the baseline and trial data
    combined_mean = np.concatenate([baseline_data, trial_data])
    combined_sem = np.concatenate([baseline_sem, sem_trial])
    x_combined = list(range(-10, len(combined_mean)-10))

    # Plot combined baseline and trial data as a continuous line
    ax2.plot(x_combined, combined_mean, label=f'{full_cond.capitalize()}',
            color=colors[cond], lw=1, markersize=4, marker='o')
    ax2.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

for axi in [ax, ax2]:

    axi.set_xlabel('Trials relative to start of feedback trials)')
    axi.legend(fontsize=14)
    axi.spines[['top', 'right']].set_visible(False)
    axi.axvline(0, ls='--', c='k', lw=1, label='Start of Task Trials')
    axi.axhline(0, ls='--', c='k', lw=1)

ax.set_title('Avg. Feedback - Confidence')
ax.set_ylabel('Avg. Feedback - Confidence\n(mean+-SEM)')
ax2.set_title('|Avg. Feedback - Confidence|')
ax2.set_ylabel('|Avg. Feedback - Confidence|\n(mean+-SEM)')

plt.tight_layout()
plt.show()


#%%
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
p_value_threshold = 0.05  # Threshold for significance in learning

fig, axes = plt.subplots(len(conditions), 2, figsize=(12, len(conditions) * 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Initialize dictionaries to store p-values and r-values for each condition
p_values_dict = {cond: [] for cond in conditions}
r_values_dict = {cond: [] for cond in conditions}

for idx, cond in enumerate(conditions):
    df_cond = df_a[df_a['condition_list'] == cond]

    for index, row in df_cond.iterrows():
        trial_data = row['feedback'] - row['confidence_task']
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(trial_data)), trial_data)
        p_values_dict[cond].append(p_value)
        r_values_dict[cond].append(r_value)

    # FDR correction
    rejected, pvals_corrected, _, _ = multipletests(p_values_dict[cond],
                                                    alpha=0.05, method='fdr_bh')

    # Plot histograms
    axes[idx, 0].hist(pvals_corrected, bins=20, color=colors[cond], alpha=0.7)
    axes[idx, 0].set_title(f'{cond.capitalize()} Condition: P-values')
    axes[idx, 0].axvline(p_value_threshold, color='red', linestyle='dashed',
                         linewidth=1)
    axes[idx, 0].set_xlabel('P-value')
    axes[idx, 0].set_ylabel('Frequency')

    axes[idx, 1].hist(r_values_dict[cond], bins=20, color=colors[cond],
                      alpha=0.7)
    axes[idx, 1].set_title(f'{cond.capitalize()} Condition: R-values')
    axes[idx, 1].set_xlabel('R-value')
    axes[idx, 1].set_ylabel('Frequency')


    # Count significant learning instances with r < 0
    significant_learning_count = sum([pval < p_value_threshold
                                      and rval < 0
                                      for pval, rval
                                      in zip(pvals_corrected,
                                             r_values_dict[cond])])
    total_count = len(df_cond)
    expected_by_chance = total_count * p_value_threshold
    binom_test_result = binomtest(significant_learning_count, total_count,
                              p_value_threshold).pvalue

    print(f"Condition {cond}: {significant_learning_count} out of {total_count} participants showed significant negative learning (p < {p_value_threshold}).")
    print(f"Expected by chance: {expected_by_chance:.2f}. Binomial test p-value: {binom_test_result:.4f}\n")


plt.show()
#%%

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import binomtest  # Use the updated binomtest function

conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
p_value_threshold = 0.05  # Threshold for significance in learning

fig, axes = plt.subplots(len(conditions), 2, figsize=(12, len(conditions) * 4))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

# Initialize dictionaries to store p-values and slopes for each condition
p_values_dict = {cond: [] for cond in conditions}
slopes_dict = {cond: [] for cond in conditions}

for idx, cond in enumerate(conditions):
    df_cond = df_a[df_a['condition_list'] == cond]

    for index, row in df_cond.iterrows():
        trial_data = row['feedback'] - row['confidence_task']
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(trial_data)), trial_data)
        p_values_dict[cond].append(p_value)
        slopes_dict[cond].append(slope)

    # FDR correction
    rejected, pvals_corrected, _, _ = multipletests(p_values_dict[cond],
                                                    alpha=0.05,
                                                    method='fdr_bh')

    # Plot histograms
    axes[idx, 0].hist(pvals_corrected, bins=20, color=colors[cond], alpha=0.7)
    axes[idx, 0].set_title(f'{cond.capitalize()} Condition: Corrected P-values')
    axes[idx, 0].axvline(p_value_threshold, color='red', linestyle='dashed',
                         linewidth=1)
    axes[idx, 0].set_xlabel('P-value')
    axes[idx, 0].set_ylabel('Frequency')

    axes[idx, 1].hist(slopes_dict[cond], bins=20, color=colors[cond],
                      alpha=0.7)
    axes[idx, 1].set_title(f'{cond.capitalize()} Condition: Slopes')
    axes[idx, 1].set_xlabel('Slope')
    axes[idx, 1].set_ylabel('Frequency')

    axes[idx, 0].spines[['top', 'right']].set_visible(False)
    axes[idx, 1].spines[['top', 'right']].set_visible(False)

    # Count significant learning instances with slope < 0
    significant_learning_count = sum([pval < p_value_threshold
                                      and slope < 0
                                      for pval, slope in
                                      zip(pvals_corrected, slopes_dict[cond])])
    total_count = len(df_cond)
    expected_by_chance = total_count * p_value_threshold
    binom_test_result = binomtest(significant_learning_count,
                                  total_count, p_value_threshold).pvalue

    print(f"Condition {cond}: {significant_learning_count} out of {total_count} participants showed significant learning (negative slope with p < {p_value_threshold}).")
    print(f"Expected by chance: {expected_by_chance:.2f}. Binomial test p-value: {binom_test_result:.4f}\n")

plt.show()

#%%

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
p_value_threshold = 0.05  # Threshold for significance in learning

# Increase the number of rows in the subplot to accommodate the individual traces plots
fig, axes = plt.subplots(len(conditions) + 1, 2, figsize=(12, (len(conditions) + 1) * 4), gridspec_kw={'height_ratios': [1] * len(conditions) + [2]})
fig.subplots_adjust(hspace=0.6, wspace=0.4)

# Process and plot data for each condition
for idx, cond in enumerate(conditions):
    df_cond = df_a[df_a['condition_list'] == cond]
    p_values = []
    slopes = []

    for index, row in df_cond.iterrows():
        trial_data = row['feedback'][:10] - row['confidence_task'][:10]
        normalized_data = trial_data - trial_data[0]  # Normalize trial data to start value
        slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(trial_data)), trial_data)
        p_values.append(p_value)
        slopes.append(slope)

        # Plot normalized individual traces on the last row subplot
        axes[-1, 0].plot(normalized_data, color=colors[cond], alpha=0.3, lw=1)

    # FDR correction
    rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Plot histograms for corrected p-values and slopes
    axes[idx, 0].hist(pvals_corrected, bins=10, color=colors[cond], alpha=0.7)
    axes[idx, 0].set_title(f'{cond.capitalize()} Condition: Corrected P-values')
    axes[idx, 0].axvline(p_value_threshold, color='red', linestyle='dashed', linewidth=1)
    axes[idx, 0].set_xlabel('P-value')
    axes[idx, 0].set_ylabel('Frequency')

    axes[idx, 1].hist(slopes, bins=10, color=colors[cond], alpha=0.7)
    axes[idx, 1].set_title(f'{cond.capitalize()} Condition: Slopes')
    axes[idx, 1].set_xlabel('Slope')
    axes[idx, 1].set_ylabel('Frequency')

    # Count significant learning instances with slope < 0
    significant_learning_count = sum(pval < p_value_threshold and slope < 0 for pval, slope in zip(pvals_corrected, slopes))
    binom_test_result = binomtest(significant_learning_count, len(df_cond), p_value_threshold).pvalue

    print(f"Condition {cond}: {significant_learning_count} out of {len(df_cond)} participants showed significant learning (negative slope with p < {p_value_threshold}). Binomial test p-value: {binom_test_result:.4f}")

# Set titles and labels for the individual traces plots
for idx, cond in enumerate(conditions):
    axes[-1, idx].set_title(f'{cond.capitalize()} Condition: Normalized Individual Traces')
    axes[-1, idx].set_xlabel('Trial')
    axes[-1, idx].set_ylabel('Normalized Difference')

plt.tight_layout()
plt.show()



#%%

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Process trial and baseline data
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]
    num_participants = len(df_cond)

    if num_participants == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = [np.mean([participant[i]
                           for participant in df_cond['p_avg_task']])
                  for i in range(20)]
    sem_trial = [np.std([participant[i]
                         for participant in df_cond['p_avg_task']],
                        ddof=1) / np.sqrt(num_participants)
                 for i in range(20)]

    # Process baseline data: Use the last 10 trials for each participant
    last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(num_participants)

    # Adjust baseline and trial data for plotting
    x_baseline = list(range(-10, 0))
    x_trial = list(range(0, 20))
    x_combined = x_baseline + x_trial

    # Combine the baseline and trial data
    combined_mean = list(baseline_data) + trial_data
    combined_sem = list(baseline_sem) + sem_trial

    # Plot combined baseline and trial data as a continuous line
    ax.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
            color=colors[cond], lw=2, marker='o')
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

ax.set_title('Relative Performance Percentile')
ax.set_xlabel('Trials relative to start of task trials')
ax.set_ylabel('Performance percentile (mean+-SEM)')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

plt.tight_layout()
plt.show()

#%%
# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}
mean_derivative_baseline = {cond: [] for cond in conditions}
sem_derivative_baseline = {cond: [] for cond in conditions}
mean_derivative_task = {cond: [] for cond in conditions}
sem_derivative_task = {cond: [] for cond in conditions}



# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 7))

# Process trial and baseline data
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]
    num_participants = len(df_cond)

    if num_participants == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = [np.mean([participant[i]
                           for participant in df_cond['p_avg_task']])
                  for i in range(20)]
    sem_trial = [np.std([participant[i]
                         for participant in df_cond['p_avg_task']],
                        ddof=1) / np.sqrt(num_participants)
                 for i in range(20)]

    # Process baseline data: Use the last 10 trials for each participant
    last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(num_participants)

    # Calculate derivatives for each participant
    derivatives_baseline = [np.mean(np.diff(participant[-10:]))
                            for participant in df_cond['p_avg_baseline']]
    derivatives_task = [np.mean(np.diff(participant))
                        for participant in df_cond['p_avg_task']]

    # Calculate mean derivative and SEM across participants
    mean_derivative_baseline[cond] = np.mean(derivatives_baseline)
    sem_derivative_baseline[cond] = np.std(derivatives_baseline,
                                           ddof=1) / np.sqrt(num_participants)
    mean_derivative_task[cond] = np.mean(derivatives_task)
    sem_derivative_task[cond] = np.std(derivatives_task,
                                       ddof=1) / np.sqrt(num_participants)


    # Adjust baseline and trial data for plotting
    x_baseline = list(range(-10, 0))
    x_trial = list(range(0, 20))
    x_combined = x_baseline + x_trial

    # Combine the baseline and trial data
    combined_mean = list(baseline_data) + trial_data
    combined_sem = list(baseline_sem) + sem_trial

    # Plot combined baseline and trial data as a continuous line
    axs[0].plot(x_combined, combined_mean, label=f'{cond.capitalize()}',
            color=colors[cond], lw=2, marker='o')
    axs[0].fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)


# Configuration for the first subplot
axs[0].set_title('Mean Performance Percentile')
axs[0].set_xlabel('Trial Number')
axs[0].set_ylabel('Performance')
axs[0].legend()
axs[0].spines[['top', 'right']].set_visible(False)
axs[0].axvline(0, c='k', ls='--')

# Error bar graph for mean derivatives
x_positions = np.arange(len(conditions))
bar_width = 0.2  # Adjusted for better visualization in errorbar plot

for i, cond in enumerate(conditions):
    axs[1].errorbar(x_positions[i] - bar_width, mean_derivative_baseline[cond],
                    yerr=sem_derivative_baseline[cond], fmt='o',
                    label='Baseline trials' if i == 0 else "", color='lightblue',
                    capsize=5)
    axs[1].errorbar(x_positions[i] + bar_width, mean_derivative_task[cond],
                    yerr=sem_derivative_task[cond], fmt='o',
                    label='Task trials' if i == 0 else "", color='salmon',
                    capsize=5)


# Configuration for the first subplot
axs[0].set_title('Performance Percentile (relative performance)')
axs[0].set_xlabel('Trials relative to start of task trials')
axs[0].set_ylabel('Performance percentile (mean+-SEM)')
axs[0].legend()
axs[0].spines[['top', 'right']].set_visible(False)
axs[0].axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configuration for the second subplot (bar graph)
axs[1].set_title('Mean Derivative of Performance Percentile')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Mean Derivative')
axs[1].set_xticks(np.arange(len(conditions)))
axs[1].set_xticklabels(conditions)
axs[1].legend()
axs[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%%

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}
mean_derivative_baseline = {cond: [] for cond in conditions}
sem_derivative_baseline = {cond: [] for cond in conditions}
mean_derivative_task = {cond: [] for cond in conditions}
sem_derivative_task = {cond: [] for cond in conditions}


# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 7))

# Process trial and baseline data
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]
    num_participants = len(df_cond)

    if num_participants == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = [np.mean([participant[i]
                           for participant in df_cond['performance_task']])
                  for i in range(20)]
    sem_trial = [np.std([participant[i]
                         for participant in df_cond['performance_task']],
                        ddof=1) / np.sqrt(num_participants)
                 for i in range(20)]

    # Process baseline data: Use the last 10 trials for each participant
    last_10_baseline = df_cond['performance_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(num_participants)

    # Calculate derivatives for each participant
    derivatives_baseline = [np.mean(np.diff(participant[-10:]))
                            for participant in df_cond['performance_baseline']]
    derivatives_task = [np.mean(np.diff(participant))
                        for participant in df_cond['performance_task']]

    # Calculate mean derivative and SEM across participants
    mean_derivative_baseline[cond] = np.mean(derivatives_baseline)
    sem_derivative_baseline[cond] = np.std(derivatives_baseline,
                                           ddof=1) / np.sqrt(num_participants)
    mean_derivative_task[cond] = np.mean(derivatives_task)
    sem_derivative_task[cond] = np.std(derivatives_task,
                                       ddof=1) / np.sqrt(num_participants)


    # Adjust baseline and trial data for plotting
    x_baseline = list(range(-10, 0))
    x_trial = list(range(0, 20))
    x_combined = x_baseline + x_trial

    # Combine the baseline and trial data
    combined_mean = list(baseline_data) + trial_data
    combined_sem = list(baseline_sem) + sem_trial

    # Plot combined baseline and trial data as a continuous line
    axs[0].plot(x_combined, combined_mean, label=f'{cond.capitalize()}',
            color=colors[cond], lw=2, marker='o')
    axs[0].fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)


# Configuration for the first subplot
axs[0].set_title('Mean Performance')
axs[0].set_xlabel('Trial Number')
axs[0].set_ylabel('Performance')
axs[0].legend()
axs[0].spines[['top', 'right']].set_visible(False)
axs[0].axvline(0, c='k', ls='--')

# Error bar graph for mean derivatives
x_positions = np.arange(len(conditions))
bar_width = 0.2  # Adjusted for better visualization in errorbar plot

for i, cond in enumerate(conditions):
    axs[1].errorbar(x_positions[i] - bar_width, mean_derivative_baseline[cond],
                    yerr=sem_derivative_baseline[cond], fmt='o',
                    label='Baseline trials' if i == 0 else "", color='lightblue',
                    capsize=5)
    axs[1].errorbar(x_positions[i] + bar_width, mean_derivative_task[cond],
                    yerr=sem_derivative_task[cond], fmt='o',
                    label='Task trials' if i == 0 else "", color='salmon',
                    capsize=5)


# Configuration for the first subplot
axs[0].set_title('Absolute Error (number of points from correct)')
axs[0].set_xlabel('Trials relative to start of task trials')
axs[0].set_ylabel('Absolute Error (mean+-SEM)')
axs[0].legend()
axs[0].spines[['top', 'right']].set_visible(False)
axs[0].axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configuration for the second subplot (bar graph)
axs[1].set_title('Mean Derivative of Absolute Error')
axs[1].set_xlabel('Condition')
axs[1].set_ylabel('Mean Derivative')
axs[1].set_xticks(np.arange(len(conditions)))
axs[1].set_xticklabels(conditions)
axs[1].legend()
axs[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

#%% mean pavg vs confidence


fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

# Scatter plot for 'neut' condition
df_plot = df_a[df_a['condition_list']=='neut']
mean_performance = [np.mean(i) for i in df_plot['p_avg_task']]
mean_confidence = [np.mean(i) for i in df_plot['confidence_task']]
axs[0].scatter(mean_performance, mean_confidence)
#axs[0].hist(mean_performance, bins=30)
axs[0].set_title('Neut Condition')
axs[0].set_xlabel('mean performance percentile')
axs[0].set_ylabel('mean confidence')
axs[0].spines[['top', 'right']].set_visible(False)

# Scatter plot for 'pos' condition
df_plot = df_a[df_a['condition_list']=='pos']
mean_performance = [np.mean(i) for i in df_plot['p_avg_task']]
mean_confidence = [np.mean(i) for i in df_plot['confidence_task']]
axs[1].scatter(mean_performance, mean_confidence)
axs[1].set_title('Pos Condition')
axs[1].set_xlabel('mean performance percentile')
axs[1].set_ylabel('mean confidence')
axs[1].spines[['top', 'right']].set_visible(False)

# Scatter plot for 'neg' condition
df_plot = df_a[df_a['condition_list']=='neg']
mean_performance = [np.mean(i) for i in df_plot['p_avg_task']]
mean_confidence = [np.mean(i) for i in df_plot['confidence_task']]
axs[2].scatter(mean_performance, mean_confidence)
axs[2].set_title('Neg Condition')
axs[2].set_xlabel('mean performance percentile')
axs[2].set_ylabel('mean confidence')
axs[2].spines[['top', 'right']].set_visible(False)


plt.tight_layout()
plt.show()


#%% plot Absolute error vs BDI

fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# Scatter plot for 'neut' condition
df_plot = df_a[df_a['condition_list']=='neut']
mean_performance = [np.mean(i) for i in df_plot['performance_task']]
axs[0].scatter(mean_performance, df_plot['bdi'])
#axs[0].hist(mean_performance, bins=30)
axs[0].set_title('Neut Condition')
axs[0].set_xlabel('Absolute error  Task Neut')
axs[0].set_ylabel('BDI Neut')
axs[0].spines[['top', 'right']].set_visible(False)

# Scatter plot for 'pos' condition
df_plot = df_a[df_a['condition_list']=='pos']
mean_performance = [np.mean(i) for i in df_plot['performance_task']]
axs[1].scatter(mean_performance, df_plot['bdi'])
axs[1].set_title('Pos Condition')
axs[1].set_xlabel('Absolute error  Task Pos')
axs[1].set_ylabel('BDI Pos')
axs[1].spines[['top', 'right']].set_visible(False)

# Scatter plot for 'neg' condition
df_plot = df_a[df_a['condition_list']=='neg']
mean_performance = [np.mean(i) for i in df_plot['performance_task']]
axs[2].scatter(mean_performance, df_plot['bdi'])
axs[2].set_title('Neg Condition')
axs[2].set_xlabel('Absolute error  Task Neg')
axs[2].set_ylabel('BDI Neg')
axs[2].spines[['top', 'right']].set_visible(False)


plt.tight_layout()
plt.show()


#%% Plot derivative vs BDI


fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Scatter plot for 'neut' condition
df_plot = df_a[df_a['condition_list']=='neut']
deriv_performance = [np.mean(np.diff(i)) for i in df_plot['performance_task']]
axs[0].scatter(deriv_performance, df_plot['bdi'])
axs[0].set_title('Neut Condition')
axs[0].set_xlabel('Performance Task Neut')
axs[0].set_ylabel('BDI Neut')
axs[0].spines[['top', 'right']].set_visible(False)

# Scatter plot for 'pos' condition
df_plot = df_a[df_a['condition_list']=='pos']
deriv_performance = [np.mean(np.diff(i)) for i in df_plot['performance_task']]
axs[1].scatter(deriv_performance, df_plot['bdi'])
axs[1].set_title('Pos Condition')
axs[1].set_xlabel('Performance Task Pos')
axs[1].set_ylabel('BDI Pos')
axs[1].spines[['top', 'right']].set_visible(False)
axs[1].set_xlim(-5, 4)

# Scatter plot for 'neg' condition
df_plot = df_a[df_a['condition_list']=='neg']
deriv_performance = [np.mean(np.diff(i)) for i in df_plot['performance_task']]
axs[2].scatter(deriv_performance, df_plot['bdi'])
axs[2].set_title('Neg Condition')
axs[2].set_xlabel('Performance Task Neg')
axs[2].set_ylabel('BDI Neg')
axs[2].spines[['top', 'right']].set_visible(False)


plt.tight_layout()
plt.show()

#%% Confidence

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0), (ax2, ax5), (ax3, ax6), (ax4, ax7)) = plt.subplots(4, 2,
                                                            figsize=(20, 15))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Process data for each condition
for cond, ax in zip(conditions, [ax5, ax6, ax7],
                            ):
    df_cond = df_a[df_a['condition_list'] == cond]

    if len(df_cond) == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['confidence_task']])
                           for i in range(20)])
    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['confidence_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Process baseline data
    last_10_baseline = df_cond['confidence_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(len(df_cond))

    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((baseline_sem, sem_trial))

    # Plot combined baseline and trial data as a continuous line
    x_combined = list(range(-10, 20))
    ax1.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
             color=colors[cond], lw=2, marker='o')
    ax1.fill_between(x_combined, combined_mean - combined_sem,
                     combined_mean + combined_sem, color=colors[cond],
                     alpha=0.2)

    # Calculate and plot the derivative of each participant's confidence
    for p_baseline, p_trial in zip(tqdm(df_cond['confidence_task']),
                                   last_10_baseline):

        baseline_array = np.array(p_baseline)
        trial_array = np.array(p_trial)
        p_data = np.concatenate((baseline_array, trial_array))

        # plot
        if cond == 'neut':
            derivative = np.gradient(p_data)
            ax2.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1, marker='o', markersize=0.2)
            ax2.axhline(0, c='k', ls='--', )

        if cond == 'pos':
            derivative = np.gradient(p_data)
            ax3.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1, marker='o', markersize=0.2)
            ax3.axhline(0, c='k', ls='--')

        if cond == 'neg':
            derivative = np.gradient(p_data)
            ax4.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1, marker='o', markersize=0.2)
            ax4.axhline(0, c='k', ls='--')

    # plot
    conf_flat = df_cond.explode('confidence_task').confidence_task.values
    if cond == 'neut':
        # Add histogram to the right
        divider = make_axes_locatable(ax2)
        ax_hist_2 = divider.append_axes("right", size="20%", pad=0.4)

        # Plot histogram on this new secondary axis
        ax_hist_2.hist(conf_flat, bins=25, orientation='horizontal',
                     color='grey', alpha=0.3)
        ax_hist_2.set_xlabel('Count')
        ax_hist_2.spines[['top', 'right']].set_visible(False)
        ax_hist_2.set_xlim()

    if cond == 'pos':
        # Add histogram to the right
        divider = make_axes_locatable(ax3)
        ax_hist_2 = divider.append_axes("right", size="20%", pad=0.4)

        # Plot histogram on this new secondary axis
        ax_hist_2.hist(conf_flat, bins=25, orientation='horizontal',
                     color='green', alpha=0.3)
        ax_hist_2.set_xlabel('Count')
        ax_hist_2.spines[['top', 'right']].set_visible(False)
        ax_hist_2.set_xlim()

    if cond == 'neg':
        # Add histogram to the right
        divider = make_axes_locatable(ax4)
        ax_hist_2 = divider.append_axes("right", size="20%", pad=0.4)

        # Plot histogram on this new secondary axis
        ax_hist_2.hist(conf_flat, bins=25, orientation='horizontal',
                     color='red', alpha=0.3)
        ax_hist_2.set_xlabel('Count')
        ax_hist_2.spines[['top', 'right']].set_visible(False)
        ax_hist_2.set_xlim()





    # Mean and sem for each participant
    means = [np.mean(participant)
             for participant in df_cond['confidence_task']]
    sems = [np.std(participant, ddof=1) / np.sqrt(len(participant))
            for participant in df_cond['confidence_task']]

    # Plotting error bars for each participant
    ax.errorbar(range(len(means)), means, yerr=sems, fmt='o',
                color=colors[cond], label=f'{cond} Condition')
    ax.set_xlabel('Participants')
    ax.set_ylabel('Confidence\n(mean+-SEM)')
    ax.spines[['top', 'right']].set_visible(False)

    # Add histogram to the right
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="20%", pad=0.4)

    # Plot histogram on this new secondary axis
    ax_hist.hist(means, bins=25, orientation='horizontal',
                 color=colors[cond], alpha=0.3)
    ax_hist.set_xlabel('Count')
    ax_hist.spines[['top', 'right']].set_visible(False)
    ax_hist.set_xlim()


# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
#ax1.set_title('Confidence Across Time')
ax1.set_xlabel('Trials relative to start of task trials')
ax1.set_ylabel('Confidence\n(mean±SEM)')
ax1.legend(loc='upper left', bbox_to_anchor=(1,1))
ax1.spines[['top', 'right']].set_visible(False)
ax1.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Trials relative to start of task trials')
ax2.set_ylabel('Confidence')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
ax2.set_ylim(0, 100)

ax3.set_xlabel('Trials relative to start of task trials')
ax3.set_ylabel('Confidence')
#ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)
ax3.axvline(0, ls='--', c='k', label='')
ax3.set_ylim(0, 100)

ax4.set_xlabel('Trials relative to start of task trials')
ax4.set_ylabel('Confidence')
#ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)
ax4.axvline(0, ls='--', c='k', label='')
ax4.set_ylim(0, 100)

plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels for x axis
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels for y axis
plt.rc('legend', fontsize=20)    # fontsize of the legend


#plt.tight_layout()
path = r"C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil\metacognition-learning\comparative_models\results\Fixed_feedback\trial_by_trial_analysis"
save_path = os.path.join(path, 'confidence_over_time.png')
plt.savefig(save_path, dpi=300)
plt.show()

#%% Relative Performance

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0), (ax2, ax5), (ax3, ax6), (ax4, ax7)) = plt.subplots(4, 2,
                                                            figsize=(15, 10))

# Process data for each condition
for cond, ax in zip(conditions, [ax5, ax6, ax7],
                            ):
    df_cond = df_a[df_a['condition_list'] == cond]

    if len(df_cond) == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['p_avg_task']])
                           for i in range(20)])
    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['p_avg_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Process baseline data
    last_10_baseline = df_cond['p_avg_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(len(df_cond))

    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((baseline_sem, sem_trial))

    # Plot combined baseline and trial data as a continuous line
    x_combined = list(range(-10, 20))
    ax1.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
             color=colors[cond], lw=2, marker='o')
    ax1.fill_between(x_combined, combined_mean - combined_sem,
                     combined_mean + combined_sem, color=colors[cond],
                     alpha=0.2)

    # Calculate and plot the derivative of each participant's confidence
    for p_baseline, p_trial in zip(tqdm(df_cond['p_avg_task']),
                                   last_10_baseline):

        baseline_array = np.array(p_baseline)
        trial_array = np.array(p_trial)
        p_data = np.concatenate((baseline_array, trial_array))

        # plot
        if cond == 'neut':
            derivative = np.gradient(p_data)
            ax2.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax2.axhline(0, c='k', ls='--')

        if cond == 'pos':
            derivative = np.gradient(p_data)
            ax3.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax3.axhline(0, c='k', ls='--')

        if cond == 'neg':
            derivative = np.gradient(p_data)
            ax4.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax4.axhline(0, c='k', ls='--')

    # Mean and sem for each participant
    means = [np.mean(participant) for participant in df_cond['p_avg_task']]
    sems = [np.std(participant, ddof=1) / np.sqrt(len(participant))
            for participant in df_cond['p_avg_task']]

    # Plotting error bars for each participant
    ax.errorbar(range(len(means)), means, yerr=sems, fmt='o',
                color=colors[cond], label=f'{cond} Condition')
    ax.set_xlabel('Participants')
    ax.set_ylabel('Relative Performance\n(mean+-SEM)')
    ax.spines[['top', 'right']].set_visible(False)

    # Add histogram to the right
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="20%", pad=0.1)

    # Plot histogram on this new secondary axis
    ax_hist.hist(means, bins=25, orientation='horizontal',
                 color=colors[cond], alpha=0.3)
    ax_hist.set_xlabel('Frequency')
    ax_hist.spines[['top', 'right']].set_visible(False)


# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
ax1.set_title('Relative Performance Across Conditions')
ax1.set_xlabel('Trials relative to start of task trials')
ax1.set_ylabel('Relative Performance (mean±SEM)')
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)
ax1.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Trials relative to start of task trials')
ax2.set_ylabel('Relative Performance')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
ax2.set_ylim(0, 100)

ax3.set_xlabel('Trials relative to start of task trials')
ax3.set_ylabel('Relative Performance')
#ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)
ax3.axvline(0, ls='--', c='k', label='')
ax3.set_ylim(0, 100)

ax4.set_xlabel('Trials relative to start of task trials')
ax4.set_ylabel('Relative Performance')
#ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)
ax4.axvline(0, ls='--', c='k', label='')
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.show()

#%% Absolute performance


# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0), (ax2, ax5), (ax3, ax6), (ax4, ax7)) = plt.subplots(4, 2,
                                                            figsize=(15, 10))

# Process data for each condition
for cond, ax in zip(conditions, [ax5, ax6, ax7],
                            ):
    df_cond = df_a[df_a['condition_list'] == cond]

    if len(df_cond) == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['performance_task']])
                           for i in range(20)])
    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['performance_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Process baseline data
    last_10_baseline = df_cond['performance_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(len(df_cond))

    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((baseline_sem, sem_trial))

    # Plot combined baseline and trial data as a continuous line
    x_combined = list(range(-10, 20))
    ax1.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
             color=colors[cond], lw=2, marker='o')
    ax1.fill_between(x_combined, combined_mean - combined_sem,
                     combined_mean + combined_sem, color=colors[cond],
                     alpha=0.2)

    # Calculate and plot the derivative of each participant's confidence
    for p_baseline, p_trial in zip(tqdm(df_cond['performance_task']),
                                   last_10_baseline):

        baseline_array = np.array(p_baseline)
        trial_array = np.array(p_trial)
        p_data = np.concatenate((baseline_array, trial_array))

        # plot
        if cond == 'neut':
            derivative = np.gradient(p_data)
            ax2.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax2.axhline(0, c='k', ls='--')

        if cond == 'pos':
            derivative = np.gradient(p_data)
            ax3.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax3.axhline(0, c='k', ls='--')

        if cond == 'neg':
            derivative = np.gradient(p_data)
            ax4.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax4.axhline(0, c='k', ls='--')

    # Mean and sem for each participant
    means = [np.mean(participant)
             for participant in df_cond['performance_task']]
    sems = [np.std(participant, ddof=1) / np.sqrt(len(participant))
            for participant in df_cond['performance_task']]

    # Plotting error bars for each participant
    ax.errorbar(range(len(means)), means, yerr=sems, fmt='o',
                color=colors[cond], label=f'{cond} Condition')
    ax.set_xlabel('Participants')
    ax.set_ylabel('Absolute Error\n(mean+-SEM)')
    ax.spines[['top', 'right']].set_visible(False)

    # Add histogram to the right
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="20%", pad=0.1)

    # Plot histogram on this new secondary axis
    ax_hist.hist(means, bins=25, orientation='horizontal', color=colors[cond],
                 alpha=0.3)
    ax_hist.set_xlabel('Frequency')
    ax_hist.spines[['top', 'right']].set_visible(False)


# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
ax1.set_title('Absolute Error Across Conditions')
ax1.set_xlabel('Trials relative to start of task trials')
ax1.set_ylabel('Absolute Error (mean±SEM)')
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)
ax1.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Trials relative to start of task trials')
ax2.set_ylabel('Absolute Error ')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
ax2.set_ylim(0, 200)

ax3.set_xlabel('Trials relative to start of task trials')
ax3.set_ylabel('Absolute Error ')
#ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)
ax3.axvline(0, ls='--', c='k', label='')
ax3.set_ylim(0, 200)

ax4.set_xlabel('Trials relative to start of task trials')
ax4.set_ylabel('Absolute Error ')
#ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)
ax4.axvline(0, ls='--', c='k', label='')
ax4.set_ylim(0, 200)

plt.tight_layout()
plt.show()


#%% dot estimate

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0), (ax2, ax5), (ax3, ax6), (ax4, ax7)) = plt.subplots(4, 2,
                                                            figsize=(20, 10))

# Process data for each condition
for cond, ax in zip(conditions, [ax5, ax6, ax7],
                            ):
    df_cond = df_a[df_a['condition_list'] == cond]

    if len(df_cond) == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['estimate_task']])
                           for i in range(20)])
    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['estimate_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Process baseline data
    last_10_baseline = df_cond['estimate_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(len(df_cond))

    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((baseline_sem, sem_trial))

    # Plot combined baseline and trial data as a continuous line
    x_combined = list(range(-10, 20))
    ax1.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
             color=colors[cond], lw=2, marker='o')
    ax1.fill_between(x_combined, combined_mean - combined_sem,
                     combined_mean + combined_sem, color=colors[cond],
                     alpha=0.2)

    # Calculate and plot the derivative of each participant's confidence
    for p_baseline, p_trial in zip(tqdm(df_cond['estimate_task']),
                                   last_10_baseline):

        baseline_array = np.array(p_baseline)
        trial_array = np.array(p_trial)
        p_data = np.concatenate((baseline_array, trial_array))

        # plot
        if cond == 'neut':
            derivative = np.gradient(p_data)
            ax2.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax2.axhline(0, c='k', ls='--')

        if cond == 'pos':
            derivative = np.gradient(p_data)
            ax3.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax3.axhline(0, c='k', ls='--')

        if cond == 'neg':
            derivative = np.gradient(p_data)
            ax4.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.1, alpha=1)
            ax4.axhline(0, c='k', ls='--')

    # Mean and sem for each participant
    means = [np.mean(participant)
             for participant in df_cond['estimate_task']]
    sems = [np.std(participant, ddof=1) / np.sqrt(len(participant))
            for participant in df_cond['estimate_task']]

    # Plotting error bars for each participant
    ax.errorbar(range(len(means)), means, yerr=sems, fmt='o',
                color=colors[cond], label=f'{cond} Condition')
    ax.set_xlabel('Participants')
    ax.set_ylabel('Dot Estimate\n(mean+-SEM)')
    ax.spines[['top', 'right']].set_visible(False)

    # Add histogram to the right
    divider = make_axes_locatable(ax)
    ax_hist = divider.append_axes("right", size="20%", pad=0.1)

    # Plot histogram on this new secondary axis
    ax_hist.hist(means, bins=25, orientation='horizontal', color=colors[cond],
                 alpha=0.3)
    ax_hist.set_xlabel('Frequency')
    ax_hist.spines[['top', 'right']].set_visible(False)


# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
ax1.set_title('Dot Estimate Across Conditions')
ax1.set_xlabel('Trials relative to start of task trials')
ax1.set_ylabel('Dot Estimate (mean±SEM)')
ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)
ax1.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Trials relative to start of task trials')
ax2.set_ylabel('Dot Estimate')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
#ax2.set_ylim(0, 200)

ax3.set_xlabel('Trials relative to start of task trials')
ax3.set_ylabel('Dot Estimate')
#ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)
ax3.axvline(0, ls='--', c='k', label='')
#ax3.set_ylim(0, 200)

ax4.set_xlabel('Trials relative to start of task trials')
ax4.set_ylabel('Dot Estimate')
#ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)
ax4.axvline(0, ls='--', c='k', label='')
#ax4.set_ylim(0, 200)

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Dot_estimate_plot.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)
plt.tight_layout()

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)


plt.show()


#%% Variance and dot

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0), (ax2, ax5), (ax3, ax6), (ax4, ax7)) = plt.subplots(4, 2,
                                                            figsize=(10, 7))

# Group the DataFrame by 'pid'
grouped = df.groupby(['pid', 'trial', 'session', 'condition'])

# Calculate variance for each group
variance_df = grouped['estimate'].var().reset_index(name='std')

# Sort values to ensure the last trial comes last
df = df.sort_values(by=['pid', 'session', 'trial'])

# Drop duplicates keeping the last trial of each session for each participant
last_trials = df.drop_duplicates(subset=['pid', 'session'],
                                 keep='last')[['pid', 'session', 'condition']]

# Rename the condition column to 'session_cond' for clarity and merging later
last_trials = last_trials.rename(columns={'condition': 'session_cond'})

# Merge 'last_trials' with 'variance_df' on 'pid' and 'session'
variance_df = pd.merge(variance_df, last_trials, on=['pid', 'session'],
                       how='left')

# Group by 'pid' and 'session' to identify trial counts
trial_counts = variance_df.groupby(['pid', 'session']).size()

# Identify sessions with more than 30 trials
sessions_over_30 = trial_counts[trial_counts > 30].index

# Filter and keep only the last 30 trials of those sessions
filtered_var_df = pd.DataFrame()  # New DataFrame with filtered sessions
for pid, session in sessions_over_30:
    # Extract the specific session data for the participant
    session_data = variance_df[(variance_df['pid'] == pid) &
                               (variance_df['session'] == session)]

    # Keep only the last 30 trials
    last_30 = session_data.tail(30)

    # Append the last 30 trials to the filtered DataFrame
    filtered_var_df = pd.concat([filtered_var_df, last_30])

# Reset trial numbers within each group
filtered_var_df['trial'] = filtered_var_df.groupby(['pid',
                                                    'session']).cumcount()


# Process data for each condition
for cond, ax in zip(conditions, [ax5, ax6, ax7]):
    df_cond = df_a[df_a['condition_list'] == cond]
    df_var_cond = filtered_var_df[filtered_var_df['session_cond'] == cond]

    if len(df_cond) == 0:
        print(f"No data available for condition {cond}")
        continue

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['estimate_task']])
                           for i in range(20)])
    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['estimate_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Process baseline data
    last_10_baseline = df_cond['estimate_baseline'].apply(lambda x: x[-10:])
    padded_baseline_array = np.array(last_10_baseline.tolist())

    # Calculate mean and SEM for baseline data
    baseline_data = np.nanmean(padded_baseline_array, axis=0)
    baseline_sem = np.nanstd(padded_baseline_array,
                             axis=0, ddof=1) / np.sqrt(len(df_cond))

    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((baseline_sem, sem_trial))

    # Plot combined baseline and trial data as a continuous line
    x_combined = list(range(-10, 20))
    ax1.plot(x_combined, combined_mean, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, marker='o')
    ax1.fill_between(x_combined, combined_mean - combined_sem,
                     combined_mean + combined_sem, color=colors[cond],
                     alpha=0.2)


    # Group data by 'pid'
    grouped = df_var_cond.groupby('pid')

    # Iterate through each group to plot
    for pid, group_data in grouped:

        # Extract 'trial' and 'variance' values for the current group
        x_vals = group_data['trial']
        y_vals = group_data['std']
        x_vals = list(range(-10, len(x_vals)-10))
        # Plot the values for the current group
        ax.plot(x_vals,
                y_vals,
                color=colors[cond], lw=0.2)

    ax.set_xlabel('Trials relative to start of task trials')
    ax.set_ylabel('Subtrial Variance')
    # ax.legend()  # Uncomment if you want a legend, but be cautious with many participants
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

    # Plot mean and sem of subtrial variance
    mean_std_trial = df_var_cond.groupby('trial')['std'].mean()
    sem_std_trial = df_var_cond.groupby('trial')['std'].sem()

    x_combined = list(range(-10, 20))
    ax0.plot(x_combined, mean_std_trial, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, marker='o')
    ax0.fill_between(x_combined, mean_std_trial - sem_std_trial,
                     mean_std_trial + sem_std_trial, color=colors[cond],
                     alpha=0.2)

    # Calculate and plot the value of each participant
    for p_baseline, p_trial in zip(tqdm(df_cond['estimate_task']),
                                   last_10_baseline):

        baseline_array = np.array(p_baseline)
        trial_array = np.array(p_trial)
        p_data = np.concatenate((baseline_array, trial_array))

        # plot
        if cond == 'neut':

            ax2.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.2, alpha=1)
            ax2.axhline(0, c='k', ls='--')

        if cond == 'pos':

            ax3.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.2, alpha=1)
            ax3.axhline(0, c='k', ls='--')

        if cond == 'neg':

            ax4.plot(x_combined, p_data, label='',
                     color=colors[cond], lw=0.2, alpha=1)
            ax4.axhline(0, c='k', ls='--')



# Hide ax0
#ax0.set_visible(False)
ax0.legend(loc='upper left', bbox_to_anchor=(1,1))
ax0.spines[['top', 'right']].set_visible(False)
ax0.axvline(0, ls='--', c='k', label='Start of Task Trials')
ax0.set_xlabel('Trials relative to start of task trials')
ax0.set_ylabel('Subtrial Variance\n(mean±SEM)')

# Configure the first subplot
#ax1.set_title('Dot Estimate Across Conditions')
ax1.set_xlabel('Trials relative to start of task trials')
ax1.set_ylabel('Dot Estimate\n(mean±SEM)')
#ax1.legend()
ax1.spines[['top', 'right']].set_visible(False)
ax1.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Trials relative to start of task trials')
ax2.set_ylabel('Dot Estimate')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
#ax2.set_ylim(0, 200)

ax3.set_xlabel('Trials relative to start of task trials')
ax3.set_ylabel('Dot Estimate')
#ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)
ax3.axvline(0, ls='--', c='k', label='')
#ax3.set_ylim(0, 200)

ax4.set_xlabel('Trials relative to start of task trials')
ax4.set_ylabel('Dot Estimate')
#ax4.legend()
ax4.spines[['top', 'right']].set_visible(False)
ax4.axvline(0, ls='--', c='k', label='')
#ax4.set_ylim(0, 200)

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Dot_estimate_plot_var.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)
plt.tight_layout()

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()

#%% Is the dot estimate variance changing with time? Test with Regression

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0)) = plt.subplots(1, 2, figsize=(12, 3))

# Group the DataFrame by 'pid'
grouped = df.groupby(['pid', 'trial', 'session', 'condition'])

# Calculate variance for each group
variance_df = grouped['estimate'].std().reset_index(name='std')

# Sort values to ensure the last trial comes last
df = df.sort_values(by=['pid', 'session', 'trial'])

# Drop duplicates keeping the last trial of each session for each participant
last_trials = df.drop_duplicates(subset=['pid', 'session'],
                                 keep='last')[['pid', 'session', 'condition']]

# Rename the condition column to 'session_cond' for clarity and merging later
last_trials = last_trials.rename(columns={'condition': 'session_cond'})

# Merge 'last_trials' with 'variance_df' on 'pid' and 'session'
variance_df = pd.merge(variance_df, last_trials, on=['pid', 'session'],
                       how='left')

# Group by 'pid' and 'session' to identify trial counts
trial_counts = variance_df.groupby(['pid', 'session']).size()

# Identify sessions with more than 30 trials
sessions_over_30 = trial_counts[trial_counts > 30].index

# Filter and keep only the last 30 trials of those sessions
filtered_var_df = pd.DataFrame()  # New DataFrame with filtered sessions
for pid, session in sessions_over_30:
    # Extract the specific session data for the participant
    session_data = variance_df[(variance_df['pid'] == pid) &
                               (variance_df['session'] == session)]

    # Keep only the last 30 trials
    last_30 = session_data.tail(30)

    # Append the last 30 trials to the filtered DataFrame
    filtered_var_df = pd.concat([filtered_var_df, last_30])

# Reset trial numbers within each group
filtered_var_df['trial'] = filtered_var_df.groupby(['pid',
                                                    'session']).cumcount()

# Process data for each condition
count1 = 2
count2 = 2
for cond in conditions:

    df_var_cond = filtered_var_df[filtered_var_df['session_cond'] == cond]

    # Filter on condition
    df_cond = df_a[df_a['condition_list'] == cond]

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['estimate_task']])
                           for i in range(20)]) # N=20 trial trials

    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['estimate_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Calculate mean and SEM for trial data
    baseline_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['estimate_baseline']])
                           for i in range(10)]) # N=10 baseline trials

    sem_baseline = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['estimate_baseline']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(10)])


    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((baseline_sem, sem_trial))



    # Plot combined baseline and trial data as a continuous line
    x_combined = np.array(list(range(-10, 20)))

    # Linear Regression on baseline - dot estimate
    d = {'y': baseline_data,
         'x': range(-10, 0)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    ax1.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax1.text(-9, 77 + count1, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Linear Regression on baseline - dot estimate
    d = {'y': trial_data,
         'x':  range(0, 20)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    # plot model prediction
    ax1.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax1.text(8, 77 + count1, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Plot data
    ax1.plot(x_combined, combined_mean, label='',
             color=colors[cond], lw=1, alpha=0.6)
    ax1.axvline(0, c='k', ls='--')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_xlabel('Trials relative to start of task trials')
    ax1.set_ylabel('Dot Estimate trendline')


    # Plot mean and sem of subtrial variance
    mean_std_trial = df_var_cond.groupby('trial')['std'].mean()
    sem_std_trial = df_var_cond.groupby('trial')['std'].sem()


    # Linear Regression on baseline - dot estimate variance
    d = {'y': mean_std_trial[:10],
         'x': range(-10, 0)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    ax0.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax0.text(-9, 32 + count2, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Linear Regression on baseline - dot estimate
    d = {'y': mean_std_trial[10:],
         'x':  range(0, 20)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    # plot model prediction
    ax0.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax0.text(8, 32 + count2, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Plot data
    ax0.plot(x_combined, mean_std_trial, label='',
             color=colors[cond], lw=1, alpha=0.6)
    ax0.axvline(0, c='k', ls='--')
    ax0.spines[['top', 'right']].set_visible(False)
    ax0.set_xlabel('Trials relative to start of task trials')
    ax0.set_ylabel('Dot Estimate Std trendline')

    count1 += 1.6
    count2 += 3.5

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Dot_estimate_regressions.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)


# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()



#%% Is the error changing due to time or feedback? Test with regressions

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax1, ax0)) = plt.subplots(1, 2, figsize=(12, 3))


def run_linear_regression(data):
    d = {'y': data, 'x': range(len(data))}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y, X)
    results = model.fit()
    return results.params[1], results.rsquared, results.f_pvalue  # returning slope, R^2, and p-value


# Process data for each condition
count1 = 2
count2 = 2
for cond in conditions:

    # Filter on condition
    df_cond = df_a[df_a['condition_list'] == cond]

    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['subtrial_error_mean_task']])
                           for i in range(20)]) # N=20 trial trials

    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['subtrial_error_mean_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Calculate mean and SEM for baseline data
    baseline_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['subtrial_error_mean_baseline']])
                           for i in range(10)]) # N=10 baseline trials

    sem_baseline = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['subtrial_error_mean_baseline']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(10)])


    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, trial_data))
    combined_sem = np.concatenate((sem_baseline, sem_trial))


    # Plot combined baseline and trial data as a continuous line
    x_combined = np.array(list(range(-10, 20)))

    # Linear Regression on baseline - dot estimate
    d = {'y': baseline_data,
         'x': range(-10, 0)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    ax1.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax1.text(-9, 32 + count1, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Linear Regression on baseline - dot estimate
    d = {'y': trial_data,
         'x':  range(0, 20)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    # plot model prediction
    ax1.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax1.text(8, 32 + count1, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Plot data
    ax1.plot(x_combined, combined_mean, label='',
             color=colors[cond], lw=1, alpha=0.6)
    ax1.axvline(0, c='k', ls='--')
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.set_xlabel('Trials relative to start of task trials')
    ax1.set_ylabel('Mean of subtrial abs error')


    # Calculate mean and SEM for baseline data std
    baseline_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['subtrial_error_std_baseline']])
                           for i in range(10)]) # N=10 baseline trials

    sem_baseline = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['subtrial_error_std_baseline']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(10)])

    # Calculate mean and SEM for task data std
    task_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_cond['subtrial_error_std_task']])
                           for i in range(20)]) # N=10 baseline trials

    sem_task = np.array([np.std([participant[i]
                                  for participant
                                  in df_cond['subtrial_error_std_task']],
                                 ddof=1) / np.sqrt(len(df_cond))
                          for i in range(20)])

    # Combine the baseline and trial data
    combined_mean = np.concatenate((baseline_data, task_data))
    combined_sem = np.concatenate((sem_baseline, sem_task))

    # Linear Regression on baseline - dot estimate variance
    d = {'y': baseline_data,
         'x': range(-10, 0)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    ax0.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax0.text(-9, 18 + count2, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Linear Regression on baseline - dot estimate
    d = {'y': task_data,
         'x':  range(0, 20)}
    df_reg = pd.DataFrame(d)
    Y = df_reg.y
    X = df_reg.x
    X = sm.add_constant(X)
    model = sm.OLS(Y,X)
    results = model.fit()
    model_pred = results.predict(X)
    rsquared = results.rsquared
    pvalue = results.f_pvalue

    # plot model prediction
    ax0.plot(X.x, model_pred, label=f'{cond.capitalize()}',
             color=colors[cond], lw=2, alpha=1)

    # Annotate R^2 and p-value
    ax0.text(8, 18 + count2, f'$R^2$ = {rsquared:.2f}\np = {pvalue:.3f}',
             fontsize=10, color=colors[cond])

    # Plot data
    ax0.plot(x_combined, combined_mean, label='',
             color=colors[cond], lw=1, alpha=0.6)
    ax0.axvline(0, c='k', ls='--')
    ax0.spines[['top', 'right']].set_visible(False)
    ax0.set_xlabel('Trials relative to start of task trials')
    ax0.set_ylabel('Std of Subtrial Abs Error')

    count1 += 2.1
    count2 += 1.6

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Subtrial_error_regressions.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)


# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()






#%% Is the mean dot estimate and variance different with or without feedback?
#   Compare means

# Create subplots
fig, ((ax1, ax0)) = plt.subplots(1, 2, figsize=(12, 3))

def draw_significance_brackets(ax, x1, x2, y, h, sig, p_value):
    """
    Draws significance brackets and the corresponding text on the plot.

    Parameters:
    - ax: The axis to draw on.
    - x1, x2: The x positions of the bracket ends.
    - y: The y position of the bracket.
    - h: The height of the bracket.
    - sig: The significance value, if True draw '*', otherwise 'ns'.
    """
    bar_loc = (x1 + x2) / 2
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    if sig:
        # Determine significance level
        if round(p_value,10) <= 0.001:
            significance = "***"
        elif round(p_value,10) <= 0.01:
            significance = "**"
        elif round(p_value,10) <= 0.05:
            significance = "*"
        else:
            significance = 'error'
            print(p_value)

        ax.text(bar_loc, y + h, significance, ha='center', va='bottom',
                color='black',
                fontsize=14)
    else:
        ax.text(bar_loc, y + h, 'ns', ha='center', va='bottom', color='black',
                fontsize=14)



# Process data for each condition
count1 = 1
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]

    # Calculate mean for each participant in baseline and task
    baseline_means = df_cond['estimate_baseline'].apply(np.mean)
    task_means = df_cond['estimate_task'].apply(np.mean)

    # Perform Wilcoxon Signed-Rank Test
    wilcoxon_test_estimate = stats.wilcoxon(baseline_means, task_means)
    print(f"{cond} - p-value:" +
          f"{wilcoxon_test_estimate.pvalue}")

    # Plotting for Dot Estimate
    for idx, (b, t) in enumerate(zip(baseline_means, task_means)):
        x = [count1,count1+1]
        y = [b,t]
        if idx == 0:  # Only label the first line
            ax1.plot(x, y, marker='o', color=colors[cond], lw=0.2,
                  label=f'{cond.capitalize()}')
        else:
            ax1.plot(x, y, marker='o', color=colors[cond], lw=0.2)

    x_ticklabels = ['baseline', 'task']*3
    ax1.set_xticks(range(1, len(x_ticklabels)+1))
    ax1.set_xticklabels(x_ticklabels)

    # Repeat for variance
    baseline_variances = df_cond['subtrial_std_baseline'].apply(np.mean)
    task_variances = df_cond['subtrial_std_task'].apply(np.mean)

    # Perform Wilcoxon Signed-Rank Test for variance
    wilcoxon_test_variance = stats.wilcoxon(baseline_variances,
                                            task_variances)
    print(f"{cond} - p-value:" +
          f"{wilcoxon_test_variance.pvalue}")

    # Plotting for Dot Estimate
    for idx, (b, t) in enumerate(zip(baseline_variances, task_variances)):
        x = np.array([count1,count1+1])
        y = np.array([b,t])
        if idx == 0:  # Only label the first line
            ax0.plot(x, y, marker='o', color=colors[cond], lw=0.2,
                  label=f'{cond.capitalize()}')
        else:
            ax0.plot(x, y, marker='o', color=colors[cond], lw=0.2)

    x_ticklabels = ['baseline', 'task']*3
    ax0.set_xticks(range(1, len(x_ticklabels)+1))
    ax0.set_xticklabels(x_ticklabels)

    # Determine the y position for brackets based on the current plot
    hight_bonus = 10 # Percentage
    max_y_estimate = max(max(baseline_means), max(task_means))
    max_y_estimate = max_y_estimate + (max_y_estimate/100 * hight_bonus)
    max_y_variance = max(max(baseline_variances), max(task_variances))
    max_y_variance = max_y_variance + (max_y_variance/100 * hight_bonus)

    # Draw brackets with significance for Dot Estimate
    draw_significance_brackets(ax1, count1, count1 + 1, max_y_estimate,
                               h=0.1 * max_y_estimate,
                               sig=wilcoxon_test_estimate.pvalue < 0.05/3,
                               p_value=wilcoxon_test_estimate.pvalue)

    # Draw brackets with significance for Variance
    draw_significance_brackets(ax0, count1, count1 + 1, max_y_variance,
                               h=0.1 * max_y_variance,
                               sig=wilcoxon_test_variance.pvalue < 0.05/3,
                               p_value=wilcoxon_test_variance.pvalue)

    count1 += 2

# Setting labels and titles for the plots
ax1.set_xlabel('Condition')
ax1.set_ylabel('Mean of subtrial estimate')
ax1.axhline(df['correct'].unique().mean(), ls='--', c='k',
            label='Correct')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.spines[['top', 'right']].set_visible(False)

ax0.set_xlabel('Condition')
ax0.set_ylabel('Std of subtrial estimate')
ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax0.spines[['top', 'right']].set_visible(False)


# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Dot_estimate_comapare_means.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

plt.tight_layout()

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()



#%% Is the error in subtrial estimates different with or without feedback?
#   Compare means

# Create subplots
fig, ((ax1, ax0)) = plt.subplots(1, 2, figsize=(12, 3))

def draw_significance_brackets(ax, x1, x2, y, h, sig, p_value):
    """
    Draws significance brackets and the corresponding text on the plot.

    Parameters:
    - ax: The axis to draw on.
    - x1, x2: The x positions of the bracket ends.
    - y: The y position of the bracket.
    - h: The height of the bracket.
    - sig: The significance value, if True draw '*', otherwise 'ns'.
    """
    bar_loc = (x1 + x2) / 2
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    if sig:
        # Determine significance level
        if round(p_value,10) <= 0.001:
            significance = "***"
        elif round(p_value,10) <= 0.01:
            significance = "**"
        elif round(p_value,10) <= 0.05:
            significance = "*"
        else:
            significance = 'error'
            print(p_value)

        ax.text(bar_loc, y + h, significance, ha='center', va='bottom',
                color='black',
                fontsize=14)
    else:
        ax.text(bar_loc, y + h, 'ns', ha='center', va='bottom', color='black',
                fontsize=14)


# Process data for each condition
count1 = 1
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]

    # Calculate mean for each participant in baseline and task
    baseline_means = df_cond['subtrial_error_mean_baseline'].apply(np.mean)
    task_means = df_cond['subtrial_error_mean_task'].apply(np.mean)

    # Mean for plot
    overall_baseline_mean = np.mean(baseline_means)
    overall_task_mean = np.mean(task_means)

    # Perform Wilcoxon Signed-Rank Test
    wilcoxon_test_estimate = stats.wilcoxon(baseline_means, task_means)
    print(f"{cond} - p-value:" +
          f"{wilcoxon_test_estimate.pvalue}")

    # Plotting for Dot Estimate
    for idx, (b, t) in enumerate(zip(baseline_means, task_means)):
        x = [count1,count1+1]
        y = [b,t]
        if idx == 0:  # Only label the first line
            ax1.plot(x, y, marker='o', color=colors[cond], lw=0.2,
                  label=f'{cond.capitalize()}')
        else:
            ax1.plot(x, y, marker='o', color=colors[cond], lw=0.2)

    x = np.array([count1,count1+1])
    y = [overall_baseline_mean, overall_task_mean]
    ax1.plot(x, y,
             marker='o', lw=1, c='k')

    x_ticklabels = ['baseline', 'task']*3
    ax1.set_xticks(range(1, len(x_ticklabels)+1))
    ax1.set_xticklabels(x_ticklabels)

    # Repeat for variance
    baseline_variances = df_cond['subtrial_error_std_baseline'].apply(np.mean)
    task_variances = df_cond['subtrial_error_std_task'].apply(np.mean)

    # Mean for plot
    overall_baseline_mean = np.mean(baseline_variances)
    overall_task_mean = np.mean(task_variances)

    # Perform Wilcoxon Signed-Rank Test for variance
    wilcoxon_test_variance = stats.wilcoxon(baseline_variances,
                                            task_variances)
    print(f"{cond} - p-value:" +
          f"{wilcoxon_test_variance.pvalue}")

    # Plotting for Dot Estimate
    for idx, (b, t) in enumerate(zip(baseline_variances, task_variances)):
        x = np.array([count1,count1+1])
        y = np.array([b,t])
        if idx == 0:  # Only label the first line
            ax0.plot(x, y, marker='o', color=colors[cond], lw=0.2,
                  label=f'{cond.capitalize()}')
        else:
            ax0.plot(x, y, marker='o', color=colors[cond], lw=0.2)

    x = np.array([count1,count1+1])
    y = [overall_baseline_mean, overall_task_mean]
    ax0.plot(x, y,
             marker='o', lw=1, c='k')

    x_ticklabels = ['baseline', 'task']*3
    ax0.set_xticks(range(1, len(x_ticklabels)+1))
    ax0.set_xticklabels(x_ticklabels)

    # Determine the y position for brackets based on the current plot
    hight_bonus = 10 # Percentage
    max_y_estimate = max(max(baseline_means), max(task_means))
    max_y_estimate = max_y_estimate + (max_y_estimate/100 * hight_bonus)
    max_y_variance = max(max(baseline_variances), max(task_variances))
    max_y_variance = max_y_variance + (max_y_variance/100 * hight_bonus)

    # Draw brackets with significance for Dot Estimate
    draw_significance_brackets(ax1, count1, count1 + 1, max_y_estimate,
                               h=0.1 * max_y_estimate,
                               sig=wilcoxon_test_estimate.pvalue < 0.05/3,
                               p_value=wilcoxon_test_estimate.pvalue)

    # Draw brackets with significance for Variance
    draw_significance_brackets(ax0, count1, count1 + 1, max_y_variance,
                               h=0.1 * max_y_variance,
                               sig=wilcoxon_test_variance.pvalue < 0.05/3,
                               p_value=wilcoxon_test_variance.pvalue)

    count1 += 2

# Setting labels and titles for the plots
ax1.set_xlabel('Condition')
ax1.set_ylabel('Mean of subtrial abs error')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.spines[['top', 'right']].set_visible(False)

ax0.set_xlabel('Condition')
ax0.set_ylabel('Std of Subtrial abs error')
ax0.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax0.spines[['top', 'right']].set_visible(False)


# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Dot_estimate_comapare_means.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

plt.tight_layout()

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()

#%% Is feedback modulating relative performance

# Create subplots
fig, (ax1) = plt.subplots(1, 1, figsize=(6, 2))

def draw_significance_brackets(ax, x1, x2, y, h, sig, p_value):
    """
    Draws significance brackets and the corresponding text on the plot.

    Parameters:
    - ax: The axis to draw on.
    - x1, x2: The x positions of the bracket ends.
    - y: The y position of the bracket.
    - h: The height of the bracket.
    - sig: The significance value, if True draw '*', otherwise 'ns'.
    """
    bar_loc = (x1 + x2) / 2
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='black')
    if sig:
        # Determine significance level
        if round(p_value,10) <= 0.001:
            significance = "***"
        elif round(p_value,10) <= 0.01:
            significance = "**"
        elif round(p_value,10) <= 0.05:
            significance = "*"
        else:
            significance = 'error'
            print(p_value)

        ax.text(bar_loc, y + h, significance, ha='center', va='bottom',
                color='black',
                fontsize=14)
    else:
        ax.text(bar_loc, y + h, 'ns', ha='center', va='bottom', color='black',
                fontsize=14)


# Process data for each condition
count1 = 1
for cond in conditions:
    df_cond = df_a[df_a['condition_list'] == cond]

    # Calculate mean for each participant in baseline and task
    baseline_means = df_cond['p_avg_baseline'].apply(np.mean)
    task_means = df_cond['p_avg_task'].apply(np.mean)

    # Mean for plot
    overall_baseline_mean = np.mean(baseline_means)
    overall_task_mean = np.mean(task_means)

    # Perform Wilcoxon Signed-Rank Test
    wilcoxon_test_estimate = stats.wilcoxon(baseline_means, task_means)
    print(f"{cond} - p-value:" +
          f"{wilcoxon_test_estimate.pvalue}")

    # Plotting for Dot Estimate
    for idx, (b, t) in enumerate(zip(baseline_means, task_means)):
        x = [count1,count1+1]
        y = [b,t]
        if idx == 0:  # Only label the first line
            ax1.plot(x, y, marker='o', color=colors[cond], lw=0.2,
                  label=f'{cond.capitalize()}')
        else:
            ax1.plot(x, y, marker='o', color=colors[cond], lw=0.2)

    x = np.array([count1,count1+1])
    y = [overall_baseline_mean, overall_task_mean]
    ax1.plot(x, y,
             marker='o', lw=1, c='k')

    x_ticklabels = ['baseline', 'task']*3
    ax1.set_xticks(range(1, len(x_ticklabels)+1))
    ax1.set_xticklabels(x_ticklabels)


    # Determine the y position for brackets based on the current plot
    hight_bonus = 10 # Percentage
    max_y_estimate = max(max(baseline_means), max(task_means))
    max_y_estimate = max_y_estimate + (max_y_estimate/100 * hight_bonus)

    # Draw brackets with significance for Dot Estimate
    draw_significance_brackets(ax1, count1, count1 + 1, max_y_estimate,
                               h=0.1 * max_y_estimate,
                               sig=wilcoxon_test_estimate.pvalue < 0.05/3,
                               p_value=wilcoxon_test_estimate.pvalue)

    count1 += 2

# Setting labels and titles for the plots
ax1.set_xlabel('Condition')
ax1.set_ylabel('Mean of\nrelative performance')
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax1.spines[['top', 'right']].set_visible(False)



# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_analysis"
else:
    result_path = r"results\variable_feedback\trial_by_trial_analysis"
file_name = 'Dot_estimate_comapare_rel_perf_means.png'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

plt.tight_layout()

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)

plt.show()

#%% Visualize why std does not need to change with lower mean error


fig, (ax,ax1) = plt.subplots(1,2, figsize=(3,3))

x1 = [1,1,1]
y1 = [1,2,4]
y1_mean = np.mean(y1)
x2 = [2,2,2]
y2 = [1,1.5,3.865]
y2_mean = np.mean(y2)
ax.scatter(x1, y1)
ax.scatter(x2, y2)
ax.plot([1,2], [y1_mean, y2_mean], c='k', marker='o')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(0.7, 2.3)
ax.set_ylabel('Mean')

y = [round(np.std(y1, ddof=1),6), round(np.std(y2, ddof=1),6)]
x = [1,2]
ax1.plot(x, y, c='k', marker='o')
ax1.spines[['top', 'right']].set_visible(False)
ax1.set_xlim(0.7, 2.3)
ax1.set_ylabel('Std')

# Disable scientific notation
ax1.ticklabel_format(useOffset=False, style='plain')

plt.tight_layout()

plt.show()


#%% Confidence and performance across trials

# Plot
fig, (ax, ax1) = plt.subplots(1,2, figsize=(6,4), sharey=True)



# Get means
for p_vals, c_vals, ax in zip(['p_avg_baseline', 'p_avg_task'],
                              ['confidence_baseline', 'confidence_task'],
                              [ax, ax1]):

    for condition, color in zip(['pos', 'neut', 'neg'],
                                ['green', 'grey', 'red']):

        if condition  == 'neut':

            df = df_a[df_a.condition_list == condition]

            p = df[p_vals].values
            c = df[c_vals].values

            # Get metacog sensitivity
            diffs = []
            for pi, ci in zip(p,c):

                diffs.append(ci-pi)

            # Compute mean and sem
            matrix = np.array(diffs)
            column_means = np.mean(matrix, axis=0)
            column_sems = stats.sem(matrix, axis=0)

            # X values (e.g., indices of the columns)
            x_values = np.arange(len(column_means))

            # Plot the mean values
            ax.plot(x_values, column_means, label='Mean', color=color)

            # Fill between the SEM values
            ax.fill_between(x_values, column_means - column_sems,
                            column_means + column_sems, color=color,
                            alpha=0.5, label='SEM')

            ax.set_xlabel('Trial')
            ax.set_ylabel('Meta Cognitive Sensitivity')
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_xticks(range(len(x_values)))
         #   ax.grid()


#%% Is confidence over trial distributed normally?


import numpy as np
import matplotlib.pyplot as plt

# Placeholder for shifted data
shifted_data = []

# Calculate the mean of each dataset and shift the data
for c_vec in df_a.confidence_task:
    mean = np.mean(c_vec)
    # Shift the data so that its mean is aligned at 0
    shifted_c_vec = c_vec - mean
    shifted_data.append(shifted_c_vec)

# Now plot all the shifted datasets on the same histogram
fig, ax
for shifted_c_vec, condition in tqdm(zip(shifted_data, df_a.condition_list),
                                     total=len(shifted_data)):
    plt.hist(shifted_c_vec, bins=8, alpha=0.5, label=condition)

plt.title('Overlayed Histograms with Aligned Means')
plt.xlabel('Shifted Value')
plt.ylabel('Frequency')
plt.axvline(0, color='k', linestyle='dashed', linewidth=1)  # Mark the common alignment point (mean)
#plt.legend()  # Show legend to identify each condition
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# Assuming there are three unique conditions
unique_conditions = np.unique(df_a.condition_list)

# Set up the subplot grid
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 columns

for ax, condition in zip(axes, unique_conditions):
    # Filter the data for the current condition
    condition_data = [c_vec for c_vec, cond in zip(df_a.confidence_task, df_a.condition_list) if cond == condition]

    # Assuming there's only one dataset per condition, if there are more, you might need another loop
    for c_vec in condition_data:
        mean = np.mean(c_vec)
        shifted_c_vec = c_vec - mean  # Shift the data to align the mean at 0

        # Plot the shifted data in the respective subplot
        ax.hist(shifted_c_vec, bins=20, alpha=0.5)
        ax.axvline(0, color='k', linestyle='dashed', linewidth=1)  # Mark the aligned mean
        ax.set_title(f'Histogram for {condition}')
        ax.set_xlabel('Shifted Value')
        ax.set_ylabel('Frequency')

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# Assuming there are three unique conditions
unique_conditions = np.unique(df_a.condition_list)
# Assign a unique x-axis center point for each condition's mean
mean_alignments = np.arange(len(unique_conditions))  # [0, 1, 2] for 3 conditions

# Create a dictionary to map each condition to its alignment point
condition_to_alignment = dict(zip(unique_conditions, mean_alignments))

# Set up the plot
plt.figure(figsize=(10, 6))

for c_vec, condition in zip(df_a.confidence_task, df_a.condition_list):
    mean = np.mean(c_vec)
    # Shift the data to align the mean at the specified point for this condition
    alignment_point = condition_to_alignment[condition]
    shifted_c_vec = c_vec - mean + alignment_point

    # Plot the shifted data
    plt.hist(shifted_c_vec, bins=8, alpha=0.5,
             label=f'{condition} (Mean aligned at {alignment_point})')

# Enhance plot
plt.title('Aligned Histograms by Condition')
plt.xlabel('Shifted Value (with unique mean alignments)')
plt.ylabel('Frequency')
plt.axvline(mean_alignments[0], color='k', linestyle='dashed', linewidth=1,
            label='Aligned Means')
for alignment in mean_alignments[1:]:
    plt.axvline(alignment, color='k', linestyle='dashed', linewidth=1)
#plt.legend()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt

# Define the colors for each condition
colors = {'pos': 'green', 'neut': 'grey', 'neg': 'red'}

# Create the figure and axis
plt.figure(figsize=(10, 6))

# Loop through each dataset and its condition
for c_vec, condition in zip(df_a.confidence_task, df_a.condition_list):
    mean = np.mean(c_vec)
    shifted_c_vec = c_vec - mean  # Shift to align the mean at 0

    # Use the condition to get the appropriate color
    color = colors.get(condition, 'blue')  # Default to 'blue' if condition is not in the dictionary

    # Plot the histogram with the specified color and label
    plt.hist(shifted_c_vec, bins=20, alpha=0.5, color=color, label=condition)

# Improve the plot
plt.axvline(0, color='k', linestyle='dashed', linewidth=1)  # Mark the aligned mean
plt.title('Aligned Histograms by Condition')
plt.xlabel('Shifted Value')
plt.ylabel('Frequency')
plt.legend({k: v for k, v in colors.items() if k in df_a.condition_list.unique()})  # Show legend with unique conditions

plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the colors for each condition
colors = {'pos': 'green', 'neut': 'grey', 'neg': 'red'}

# Get the unique conditions
unique_conditions = np.unique(df_a.condition_list)

# Set up the subplot grid
fig, axes = plt.subplots(len(unique_conditions), 1,
                         figsize=(10, 6 * len(unique_conditions)),
                         sharex=True)

# Check if there's only one subplot (in case of a single condition)
# and convert axes to an array
if len(unique_conditions) == 1:
    axes = [axes]

# Loop through each subplot/condition
for ax, condition in tqdm(zip(axes, unique_conditions),
                          total=len(unique_conditions)):
    # Filter the data for the current condition
    condition_data = [c_vec for c_vec, cond in zip(df_a.confidence_task,
                                                   df_a.condition_list)
                      if cond == condition]

    # Loop through the datasets for the current condition
    for c_vec in condition_data:
        mean = np.mean(c_vec)
        shifted_c_vec = c_vec - mean  # Shift the data so the mean is at 0

        # Plot the KDE for the shifted data in the respective subplot
        sns.kdeplot(shifted_c_vec, ax=ax, color=colors[condition],
                    label=f'{condition} (mean aligned)', lw=2)

        # Mark the aligned mean
        ax.axvline(0, color='k', linestyle='dashed', linewidth=1)
        ax.set_title(f'Distribution Outline for {condition}')
        ax.set_xlabel('Shifted Value')
        ax.set_ylabel('Density')
       # ax.legend()

plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the color scheme
colors = {'pos': 'green', 'neut': 'grey', 'neg': 'red'}

# Create a figure
plt.figure(figsize=(10, 6))

# Loop through each condition and plot the KDE
for condition in tqdm(np.unique(df_a.condition_list),
                      total=len(np.unique(df_a.condition_list))):
    # Filter the data for the current condition
    condition_data = [c_vec for c_vec, cond in
                      zip(df_a.confidence_task, df_a.condition_list)
                      if cond == condition]

    # Assuming there's only one dataset per condition,
    # if there are more, you might need another loop
    for c_vec in condition_data:
        mean = np.mean(c_vec)
        shifted_c_vec = c_vec - mean  # Shift the data so the mean is at 0

        # Plot the KDE for the shifted data
        sns.kdeplot(shifted_c_vec, color=colors[condition],
                    #label=f'{condition} (mean aligned)',
                    lw=0.1,
                    alpha=0.8)

# Mark the aligned mean
#plt.axvline(0, color='k', linestyle='dashed', linewidth=1)
plt.title('Distribution Outlines for Each Condition')
plt.xlabel('Shifted Value')
plt.ylabel('Density')
#plt.legend()
plt.show()



#%% Regression testing best predictor of confidence




for participant in df_a.pid.unique:

    df_p = df_a[df_a.pid == participant]

    # Loop over sessions
    for confidence in df_p.confidence_task:
        # Regression predicting confidence from:
        # Performance delta, feedback, previous_choice
        performance = df_a.error_task - df_a.error_task.shift(1)

#%%

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the color scheme
colors = {'pos': 'green', 'neut': 'grey', 'neg': 'red'}

# Unique conditions
unique_conditions = np.unique(df_a.condition_list)

# Set up the subplot grid
fig, axes = plt.subplots(1, 1, figsize=(6, 4))

# Loop through each condition and its corresponding axis
for ax, condition in zip([axes, axes, axes], unique_conditions):
    # Aggregate all confidence values for the current condition
    all_confidence_values = np.concatenate([c_vec
                                            for c_vec, cond in
                                            zip(df_a.confidence_task,
                                                df_a.condition_list)
                                            if cond == condition])
    # Plot the KDE for the shifted data
    sns.kdeplot(all_confidence_values, ax=ax,
                color=colors[condition], lw=2, alpha=0.8)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Density')
    ax.set_xlim(0,100)
    ax.spines[['top', 'right']].set_visible(False)

   # ax.axvline(0, color='k', linestyle='dashed', linewidth=1)  # Mark the aligned mean

plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the color scheme
colors = {'pos': 'green', 'neut': 'grey', 'neg': 'red'}

# Unique conditions
unique_conditions = np.unique(df_a.condition_list)

# Create a figure
fig, ax = plt.subplots(1,1, figsize=(6, 4))

# Loop through each condition
for condition in unique_conditions:
    # Initialize a list to hold all shifted values for the current condition
    all_shifted_values = []

    # Filter and shift the data for the current condition
    for c_vec, cond in zip(df_a.confidence_task, df_a.condition_list):
        if cond == condition:
            mean = np.mean(c_vec)
            shifted_c_vec = c_vec - mean  # Shift the data so the mean is at 0
            all_shifted_values.extend(shifted_c_vec)  # Add to the combined distribution

    # Plot the KDE for the combined shifted data of the current condition
    sns.kdeplot(all_shifted_values, color=colors[condition],
                label=f'{condition} (mean aligned)', lw=2)


ax.set_xlabel('Mean Shifted Confidence')
ax.set_ylabel('Density')
ax.spines[['top', 'right']].set_visible(False)

# Adjust the x-axis to show the original range of 0-100
plt.xlim(-100, 100)

plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np

colors = {'pos': 'green', 'neut': 'grey', 'neg': 'red'}

# Create three subplots (1 row, 3 columns) with a reasonable figsize
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Ensure df_a is defined and has the necessary columns
for ax, cond in zip(axes, ['neut', 'pos', 'neg']):
    # Filter the DataFrame for the current condition
    df_plot = df_a[df_a['condition_list'] == cond]

    # Calculate the mean of the baseline confidence for each entry
    baseline_means = df_plot['confidence_baseline'].apply(np.mean)

    # Iterate through each row in df_plot to calculate aligned_task_conf
    for i, row in df_plot.iterrows():
        # Extract the task confidence for the current row
        task_conf = row['confidence_task']

        # Calculate aligned_task_conf by subtracting the baseline mean from task_conf
        aligned_task_conf = task_conf - baseline_means.loc[i]

        # Plot the aligned_task_conf on the current axis
        ax.plot(aligned_task_conf, color=colors[cond], alpha=0.1)

    ax.set_title(cond.capitalize())  # Set the title for each subplot to the condition name

plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors

# Colors for different clusters
colors = ['blue', 'orange', 'green', 'red', 'cyan']
m_colors = ["#ffcccc", "#ff0000"]  # Light red to dark red
n_bins = 50  # Number of bins or levels in the color scale
cmap_name = "custom_red"
red_cmap = LinearSegmentedColormap.from_list(cmap_name, m_colors, N=n_bins)

# Create a figure with 1 row and 3 columns of subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

fontsize = 16
# Iterate over each condition and its corresponding axis
for ax, cond in zip(axes, ['pos', 'neg', 'neut',]):

    # Fill missing bdi scores
    df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

    # Filter the DataFrame for the current condition
    df_plot = df_a[df_a['condition_list'] == cond]

    # Calculate the mean of the baseline confidence for each entry
    baseline_means = np.mean(df_plot['performance_baseline'].values.tolist(),
                             axis=1)

    # Align the start of each task_conf trace to the mean of its baseline
    aligned_task_conf = (df_plot['p_avg_task'].values.tolist()
                         - baseline_means[:, np.newaxis])

    # Get bdi
    bdi = df_plot.bdi.values
    norm = mcolors.Normalize(vmin=np.min(bdi), vmax=np.max(bdi))
    colormap = plt.cm.Reds
    bdi_colors = colormap(norm(bdi))

    # Perform spectral clustering on the aligned task confidence data
    n_clusters = 3
    sc = SpectralClustering(n_clusters=n_clusters,
                            affinity='nearest_neighbors')
    labels = sc.fit_predict(aligned_task_conf)

    # Plot each cluster in a separate color
    for cluster in range(n_clusters):
        cluster_indices = labels == cluster
        true_indices = np.where(cluster_indices)[0]
        for cluster_idx in true_indices:
            ax.plot(range(len(aligned_task_conf[0])),
                       aligned_task_conf[cluster_idx],
                       color=colors[cluster],#bdi_colors[cluster_idx]
                      # cmap=red_cmap,
                       label=f'Cluster {cluster+1}',
                       alpha=0.6)

    if cond == 'neg':
        cond_title = 'Negative Feedback'
    elif cond == 'pos':
        cond_title = 'Positive Feedback'
    elif cond == 'neut':
        cond_title = 'Neutral Feedback'
    ax.set_title(f'{cond_title}', fontsize=fontsize)
    ax.set_xticks(range(0, 20, 2))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticklabels(range(1, 21, 2))
    ax.set_xlabel('Trials', fontsize=fontsize)
    ax.set_ylabel('Confidence\n(shifted to baseline mean)',
                  fontsize=fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    #ax.legend()


plt.tight_layout()
plt.show()


# As 0 represent the baseline mean, feedback does not appreat to affect
# confidence much in some participants


#%% Cluster on participant

# Colors for different clusters
colors = ['blue', 'orange', 'green', 'red', 'cyan', 'yellow', 'black']
m_colors = ["#ffcccc", "#ff0000"]  # Light red to dark red
n_bins = 50  # Number of bins or levels in the color scale
cmap_name = "custom_red"
red_cmap = LinearSegmentedColormap.from_list(cmap_name, m_colors, N=n_bins)



# Initialize df_cross_cond if it's not already defined
df_cross_cond = pd.DataFrame({'pid': [],
                              'aligned_task_conf': [],
                              'bdi': [],
                              }).set_index('pid')

for ax, cond in zip(axes, ['neut', 'pos', 'neg']):
    # Fill missing bdi scores
    df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

    # Filter the DataFrame for the current condition
    df_plot = df_a[df_a['condition_list'] == cond]

    # Calculate the mean of the baseline confidence for each entry
    baseline_means = np.mean(df_plot['confidence_baseline'].values.tolist(), axis=1)

    # Align the start of each task_conf trace to the mean of its baseline
    aligned_task_conf = np.array(df_plot['confidence_task'].values.tolist()) - baseline_means[:, np.newaxis]

    for pid, new_vector, bdi in zip(df_plot['pid'],
                                    aligned_task_conf, df_plot.bdi):
        # If pid exists in df_cross_cond, extend the existing aligned_task_conf
        if pid in df_cross_cond.index:
            df_cross_cond.at[pid, 'aligned_task_conf'] += list(new_vector)
            df_cross_cond.at[pid, 'bdi'] = bdi
        else:
            # If pid does not exist, create a temporary DataFrame for the new row
            temp_df = pd.DataFrame({'aligned_task_conf': [list(new_vector)],
                                    'bdi': bdi},
                                   index=[pid])
            # Use pd.concat to add the new row to df_cross_cond
            df_cross_cond = pd.concat([df_cross_cond, temp_df])

# Reset index if you want 'pid' as a column
df_cross_cond = df_cross_cond.reset_index()

# Filter out rows where aligned_task_conf vector is below 60 elements long
df_cross_cond = df_cross_cond[
                    df_cross_cond['aligned_task_conf'].apply(
                                                    lambda x: len(x) >= 60)]


concat_conf_task = np.array([i for i in
                             df_cross_cond.aligned_task_conf.values])

# Perform spectral clustering on the aligned task confidence data
n_clusters = 7
sc = SpectralClustering(n_clusters=n_clusters,
                        affinity='nearest_neighbors')
labels = sc.fit_predict(concat_conf_task)



# Plot each cluster in a separate color

# Create a figure with 1 row and 3 columns of subplots
fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True)
fontsize = 16

for cluster in range(n_clusters):
    cluster_indices = labels == cluster
    true_indices = np.where(cluster_indices)[0]
    for cluster_idx in true_indices:
        ax.plot(range(len(concat_conf_task[0])),
                   concat_conf_task[cluster_idx],
                   color=colors[cluster],#bdi_colors[cluster_idx]
                  # cmap=red_cmap,
                   label=f'Cluster {cluster+1}',
                   alpha=0.4)

if cond == 'neg':
    cond_title = 'Negative Feedback'
elif cond == 'pos':
    cond_title = 'Positive Feedback'
elif cond == 'neut':
    cond_title = 'Neutral Feedback'

#ax.set_title(f'{cond_title}', fontsize=fontsize)
#ax.set_xticks(range(0, 20, 2))
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.axvline(x=[20], c='k', ls='--')
ax.axvline(x=[40], c='k', ls='--')
#ax.set_xticklabels(range(1, 21, 2))
ax.set_xlabel('Trials', fontsize=fontsize)
ax.set_ylabel('Confidence\n(shifted to baseline mean)',
              fontsize=fontsize)
ax.spines[['top', 'right']].set_visible(False)
    #ax.legend()

plt.tight_layout()
plt.show()


#%%

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering

# Assuming concat_conf_task and df_cross_cond are already defined
n_clusters = 9
sc = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
labels = sc.fit_predict(concat_conf_task)

# Define colors for each cluster - adjust or add more colors as needed
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'yellow', 'black']

# BDI Colors
bdi = df_cross_cond.bdi.values
norm = mcolors.Normalize(vmin=np.min(bdi), vmax=np.max(bdi))
colormap = plt.cm.Reds
bdi_colors = colormap(norm(bdi))

# Create a figure with a grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharey=True)  # Adjust subplot grid if needed
axes = axes.flatten()  # Flatten the axes array for easy iteration
fontsize = 16

for cluster in range(n_clusters):
    cluster_indices = labels == cluster
    true_indices = np.where(cluster_indices)[0]
    ax = axes[cluster]  # Select the corresponding subplot for the current cluster

    for cluster_idx in true_indices:
        ax.plot(range(len(concat_conf_task[cluster_idx])),
                concat_conf_task[cluster_idx],
                color= bdi_colors[cluster_idx], #colors[cluster] # Use a color from the predefined list
                label=f'Cluster {cluster + 1}' if cluster_idx == true_indices[0] else "",  # Label for legend
                alpha=0.6)

    # Set titles and labels for each subplot
    ax.set_title(f'Cluster {cluster + 1}', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.axvline(x=20, c='k', ls='--')
    ax.axvline(x=40, c='k', ls='--')
    ax.axhline(y=0, c='k', ls='-')
    ax.set_xlabel('Trials', fontsize=fontsize)
    ax.set_ylabel('Confidence\n(aligned to baseline)', fontsize=fontsize)
    ax.spines[['top', 'right']].set_visible(False)


# Hide any unused subplots
for i in range(n_clusters, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()

#%%

import matplotlib.pyplot as plt
import numpy as np
import hdbscan
from scipy.signal import savgol_filter

# Assuming concat_conf_task and df_cross_cond are already defined

# first smooth the traces
# yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3
window_length = 3 # Must be odd
poly_order = 2
concat_conf_task_smoothed = savgol_filter(concat_conf_task,
                                          window_length,
                                          poly_order)
plt.plot(concat_conf_task_smoothed[0])


# Perform HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, gen_min_span_tree=True,
                            )
labels = clusterer.fit_predict(concat_conf_task_smoothed)

# Determine the number of clusters (excluding noise if present)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# Define colors for each cluster - you might need to adjust or add more colors as needed
colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'orange', 'yellow', 'purple', 'brown', 'pink']
if n_clusters > len(colors):
    raise ValueError("More clusters than defined colors. Please add more colors.")

# Create a figure with a grid of subplots
fig, axes = plt.subplots((n_clusters + 2) // 3, min(n_clusters, 3), figsize=(15, 10), sharey=True)
if n_clusters == 1:
    axes = [axes]
elif n_clusters <= 3:
    axes = axes.flatten()[:n_clusters]
else:
    axes = axes.flatten()
fontsize = 16

for cluster in range(n_clusters):
    if cluster == -1:
        continue  # Skip plotting for noise if present
    cluster_indices = labels == cluster
    true_indices = np.where(cluster_indices)[0]
    ax = axes[cluster]  # Select the corresponding subplot for the current cluster

    for cluster_idx in true_indices:
        ax.plot(range(len(concat_conf_task_smoothed[cluster_idx])),
                concat_conf_task_smoothed[cluster_idx],
                color=colors[cluster % len(colors)],  # Cycle through colors if not enough
                label=f'Cluster {cluster + 1}' if cluster_idx == true_indices[0] else "",  # Label for legend
                alpha=0.4)

    # Set titles and labels for each subplot
    ax.set_title(f'Cluster {cluster + 1}', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.axvline(x=20, c='k', ls='--')
    ax.axvline(x=40, c='k', ls='--')
    ax.axhline(y=0, c='k', ls='-')
    ax.set_xlabel('Trials', fontsize=fontsize)
    ax.set_ylabel('Confidence (shifted to baseline mean)', fontsize=fontsize)
    ax.spines[['top', 'right']].set_visible(False)

# Hide any unused subplots
for i in range(n_clusters, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


#%% If the feedback-indifferent group is excluded,
#   does alpha predict bdi?


# Create a figure with 1 row and 3 columns of subplots
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

fontsize = 16
# Iterate over each condition and its corresponding axis
for ax, cond in zip(axes, ['neut', 'pos', 'neg']):

    # Fill missing bdi scores
    df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

    # Filter the DataFrame for the current condition
    df_plot = df_a[df_a['condition_list'] == cond]

    # Calculate the mean of the baseline confidence for each entry
    baseline_means = np.mean(df_plot['confidence_baseline'].values.tolist(),
                             axis=1)

    task_means = np.mean(df_plot['confidence_task'].values.tolist(),
                             axis=1)
    # Get BDI
    bdi = df_plot.bdi.values
    norm = mcolors.Normalize(vmin=np.min(bdi), vmax=np.max(bdi))
    colormap = plt.cm.Reds
    bdi_colors = colormap(norm(bdi))

    # plot
    ax.scatter(baseline_means, task_means, c=bdi_colors)
    ax.plot(range(80),(range(80)), c='k')

    if cond == 'neg':
        cond_title = 'Negative Feedback'
    elif cond == 'pos':
        cond_title = 'Positive Feedback'
    elif cond == 'neut':
        cond_title = 'Neutral Feedback'

    ax.set_title(f'{cond_title}', fontsize=fontsize)
  #  ax.set_xticks(range(0, 20, 2))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
  #  ax.set_xticklabels(range(1, 21, 2))
    ax.set_xlabel('baseline', fontsize=fontsize)
    ax.set_ylabel('task',
                  fontsize=fontsize)
    ax.spines[['top', 'right']].set_visible(False)
        # Get bdi

plt.tight_layout()
plt.show()


#%% Does performance-feedback difference shrink with trials?



df_plot = df_a[df_a.condition_list == 'neut']
conf = df_plot.confidence_task.values
abs_error = df_plot.performance_task.values # absolute error

estimation_accuracy = []
for c, ae in zip(conf, abs_error #df_plot.p_avg_task.values,
                 ):
    diff = c-ae
    estimation_accuracy.append(diff)

fig, ax = plt.subplots(1,1, figsize=(6,4))

for i in estimation_accuracy:
    ax.plot(i, alpha=0.4,)

ax.plot(np.median(np.array(estimation_accuracy), axis=0), color='k', lw=2)


plt.show()


#%% Does mean confidence predict BDI?
df_plot = df_a[df_a.condition_list == 'neut']
conf_mean = [np.mean(i) for i in df_plot.confidence_task.values]
bdi = df_plot.bdi.values


plt.scatter(bdi, conf_mean)




#%% Is BDI correlated with Mean Confidence? Yes

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

for data, label in zip([df_a.confidence_baseline.values,
                        df_a.confidence_task.values],
                       ['Baseline', 'Task']):
    # Assuming df_a is a predefined DataFrame and df_plot is filtered as shown
    #df_plot = df_a#[df_a.condition_list == 'pos']
    conf_mean = [np.mean(i) for i in data]

    bdi = df_plot.bdi.values



    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(bdi, conf_mean)

    # Calculate the regression values
    regression_values = intercept + slope * bdi

    # Plot the original data
    plt.scatter(bdi, conf_mean, label=f'{label}', alpha=0.3)

    # Plot the regression line
    plt.plot(bdi, regression_values, #color='red',
             label=f'{label}')

    plt.xlabel('BDI')
    plt.ylabel('Mean Confidence')
    plt.legend()
plt.show()

# Optionally, print the regression parameters
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f'p-value: {p_value}')


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

for c_data, f_data, p_data, label in zip([df_a.confidence_task.values]
                                         [df_a.feedback.values],
                                         [df_a.performance_task.values]

                                         ['label']):

    for participant in range(c_data):
        c_data[participant]
        f_data[participant]
        p_data[participant]


plt.show()

# Optionally, print the regression parameters
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f'p-value: {p_value}')


#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

for data, label in zip([df_a.a_empirical.values],
                       ['Empirical Learning rate']):
    # Assuming df_a is a predefined DataFrame and df_plot is filtered as shown
    #df_plot = df_a#[df_a.condition_list == 'pos']
    a_emp_mean = np.array([np.median(i) for i in data])
    indices_over_0 = [index for index, value in enumerate(a_emp_mean)
                      if value > -100]

    bdi = df_a.bdi.values

    # filter on positive alpha
    a_emp_mean = a_emp_mean[indices_over_0]
    bdi = bdi[indices_over_0]

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(bdi, a_emp_mean)

    # Calculate the regression values
    regression_values = intercept + slope * bdi

    # Plot the original data
    plt.scatter(bdi, a_emp_mean, label=f'{label}', alpha=0.3)
    # plt.ylim(-1,4)

    # Plot the regression line
    plt.plot(bdi, regression_values, color='red',
             label=f'Trend line')

   # plt.xlim(-10,100)
    #plt.xlabel('BDI')
    #plt.ylabel('Mean Empirical Learning rate')
    plt.legend()

plt.show()

# Optionally, print the regression parameters
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f'p-value: {p_value}')


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

# Assuming 'condition_list' column exists and contains session conditions
conditions = df_a['condition_list'].unique()

# Initialize dictionaries to store the count of participants with significant predictors per condition
significant_counts_feedback = {condition: 0 for condition in conditions}
significant_counts_performance = {condition: 0 for condition in conditions}

total_participants_per_condition = {condition: 0 for condition in conditions}

# Init dictionaries for condition
sig_pval_f_pid = {condition: [] for condition in conditions}
sig_pval_p_pid = {condition: [] for condition in conditions}

# Loop through each condition
for condition in conditions:
    condition_data = df_a[df_a['condition_list'] == condition]
    participant_ids = condition_data['pid'].unique()

    # Update total participants count for the condition
    total_participants_per_condition[condition] += len(participant_ids)

    count=0
    # Loop through each participant in the current condition
    for participant_id in participant_ids:
        participant_data = condition_data[condition_data['pid'] == participant_id]


        # Get data for the current participant
        c_data = participant_data['confidence_task'].values[0][1:] # Current
        f_data = participant_data['feedback'].values[0][:-1] # Previous
        p_data = participant_data['performance_task'].values[0][1:] # Current

        # Check if there's enough data to perform regression
        if len(c_data) > 1 and len(set(f_data)) > 1:

            # Perform regression for feedback data
            _, _, _, p_value_f, _ = stats.linregress(f_data, c_data)

            # Perform regression for performance data
            _, _, _, p_value_p, _ = stats.linregress(p_data, c_data)

            # Check if the predictors are significant and update the count
            if p_value_f < 0.05:
                significant_counts_feedback[condition] += 1
                sig_pval_f_pid[condition].append(participant_data.pid.values[0])
            if p_value_p < 0.05:
                significant_counts_performance[condition] += 1
                sig_pval_p_pid[condition].append(participant_data.pid.values[0])

        else:
            count += 1

    print(count, f'{condition} - constant feedback')

# Calculate percentages
percentage_significant_feedback = {condition:
                                   (significant_counts_feedback[condition] / total_participants_per_condition[condition]) * 100 for condition in conditions}
percentage_significant_performance = {condition:
                                     (significant_counts_performance[condition] / total_participants_per_condition[condition]) * 100 for condition in conditions}

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(8, 5), sharey=True)

# Histogram for feedback data significance percentages across conditions
axs[0].bar(percentage_significant_feedback.keys(),
           percentage_significant_feedback.values(),
           color='skyblue', edgecolor='black')
axs[0].set_xlabel('Condition')
axs[0].set_ylabel('Percentage of Participants')

# Histogram for performance data significance percentages across conditions
axs[1].bar(percentage_significant_performance.keys(),
           percentage_significant_performance.values(),
           color='lightgreen', edgecolor='black')
axs[1].set_xlabel('Condition')

# Remove spines top and right
for ax in axs:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

# Filter on cond
cond = 'pos'
df_plot = df_a[df_a.condition_list == cond]

# Filter on
df_plot = df_plot[df_plot['pid'].isin(sig_pval_f_pid[cond])]

# Get empirical alpha
a_emp_mean = np.array([np.median(i) for i in df_plot.a_empirical])
indices_over_treshold = [index for index, value in enumerate(a_emp_mean)
                         if value > -100]

# Get bdi
bdi = df_plot.bdi.values

# filter on positive alpha
a_emp_mean = a_emp_mean[indices_over_treshold]
bdi = bdi[indices_over_treshold]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(bdi, a_emp_mean)

# Calculate the regression values
regression_values = intercept + slope * bdi


# Plot the original data
plt.scatter(bdi, a_emp_mean, label=f'{label}', alpha=0.3)
# plt.ylim(-1,4)

# Plot the regression line
plt.plot(bdi, regression_values, color='red',
         label=f'Trend line')

# plt.xlim(-10,100)
# plt.xlabel('BDI')
# plt.ylabel('Mean Empirical Learning rate')
plt.legend()

plt.show()

# Optionally, print the regression parameters
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
print(f"R-squared: {r_value**2}")
print(f'p-value: {p_value}')

#%% Does BDI predict delta confidence in each condition?










