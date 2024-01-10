# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:33:30 2023

@author: carll
"""

# Analysis

import numpy as np
from src.utility_functions import (add_session_column)
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import mannwhitneyu

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

subtrial_error_mean_baseline_list = []
subtrial_error_variance_baseline_list = []
subtrial_error_std_baseline_list = []
subtrial_error_mean_task_list = []
subtrial_error_variance_task_list = []
subtrial_error_std_task_list = []


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

        # Self-estimate error

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

                     })



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
                           for participant in df_cond['confidence_task']])
                  for i in range(20)]
    sem_trial = [np.std([participant[i]
                         for participant in df_cond['confidence_task']], ddof=1) / np.sqrt(num_participants)
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
    ax.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
            color=colors[cond], lw=2, marker='o')
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

ax.set_title('Mean Confidence Across Conditions')
ax.set_xlabel('Trials relative to start of task trials')
ax.set_ylabel('Confidence (mean+-SEM)')
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
    ax_hist = divider.append_axes("right", size="20%", pad=0.1)

    # Plot histogram on this new secondary axis
    ax_hist.hist(means, bins=25, orientation='horizontal',
                 color=colors[cond], alpha=0.3)
    ax_hist.set_xlabel('Frequency')
    ax_hist.spines[['top', 'right']].set_visible(False)


# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
ax1.set_title('Confidence Across Conditions')
ax1.set_xlabel('Trials relative to start of task trials')
ax1.set_ylabel('Confidence (mean±SEM)')
ax1.legend()
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

plt.tight_layout()
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






