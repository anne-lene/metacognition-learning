# -*- coding: utf-8 -*-
"""
Created on Mon May  6 18:40:49 2024

@author: carll
"""

# Analysis of varied feedback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from src.utility_functions import add_session_column
from mpl_toolkits.axes_grid1 import make_axes_locatable
from src.models import (fit_model,
                        fit_random_model,
                        random_model_w_bias,
                        win_stay_lose_shift,
                        rw_symmetric_LR,
                        rw_cond_LR,
                        choice_kernel,
                        RW_choice_kernel,
                        delta_P_RW
                        )

# Import data - Varied feedback condition (Experiment 2)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

# Add session column
df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

#%%
# Add bdi column
df['bdi'] = df['bdi_score']

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

       # p_avg_task = df_s_task.pavg.values
        #p_avg_baseline = df_s_baseline.pavg.values

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
       # p_avg_task_list.append(p_avg_task)
        #p_avg_baseline_list.append(p_avg_baseline)
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
                    # 'p_avg_task': p_avg_task_list,
                     #'p_avg_baseline': p_avg_baseline_list,
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
df_a.to_csv('varied_feedback_data.csv')

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Split data in high and low bdi
df_a['bdi_level'] = np.where(df_a.bdi.values < df_a.bdi.median(), 'low', 'high')


#%%

# Set df to df_a
df = df_a.copy()

# Initialize plot variables
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}

# Create subplots
fig, ((ax, ax0), (ax2, ax5))  = plt.subplots(2, 2,figsize=(20, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Process data for each session

# Calculate mean and SEM for trial data
trial_data = np.array([np.mean([participant[i]
                                for participant
                                in df['confidence_task']])
                       for i in range(15)])
sem_trial = np.array([np.std([participant[i]
                              for participant
                              in df['confidence_task']],
                             ddof=1) / np.sqrt(len(df))
                      for i in range(15)])


# Combine the baseline and trial data
combined_mean = trial_data
combined_sem = sem_trial

# Plot combined baseline and trial data as a continuous line
x_combined = list(range(0, 15))
ax.plot(x_combined, combined_mean, label='Confidence',
         color='k', lw=2, marker='o')
ax.fill_between(x_combined, combined_mean - combined_sem,
                 combined_mean + combined_sem, color='k',
                 alpha=0.2)

# Calculate and plot the derivative of each participant's confidence
for p_trial in tqdm(df['confidence_task'], total=len(df['confidence_task'])):

    trial_array = np.array(p_trial)
    p_data = trial_array # np.concatenate((baseline_array, trial_array))

    # plot
    derivative = np.gradient(p_data)
    ax2.plot(x_combined, p_data, label='',
             color='k', lw=0.1, alpha=1, marker='o', markersize=0.2)
    ax2.axhline(0, c='k', ls='--', )


# plot
conf_flat = df.explode('confidence_task').confidence_task.values

# Add histogram to the right
divider = make_axes_locatable(ax2)
ax_hist_2 = divider.append_axes("right", size="20%", pad=0.4)

# Plot histogram on this new secondary axis
ax_hist_2.hist(conf_flat, bins=25, orientation='horizontal',
             color='grey', alpha=0.3)
ax_hist_2.set_xlabel('Count')
ax_hist_2.spines[['top', 'right']].set_visible(False)
ax_hist_2.set_xlim()


# Mean and sem for each participant
means = [np.mean(participant)
         for participant in df['confidence_task']]
sems = [np.std(participant, ddof=1) / np.sqrt(len(participant))
        for participant in df['confidence_task']]

# Plotting error bars for each participant
ax5.errorbar(range(len(means)), means, yerr=sems, fmt='o',
            color='k',)
ax5.set_xlabel('Participants')
ax5.set_ylabel('Confidence\n(mean+-SEM)')
ax5.spines[['top', 'right']].set_visible(False)

# Add histogram to the right
divider = make_axes_locatable(ax5)
ax_hist = divider.append_axes("right", size="20%", pad=0.4)

# Plot histogram on this new secondary axis
ax_hist.hist(means, bins=25, orientation='horizontal',
             color='k', alpha=0.3)
ax_hist.set_xlabel('Count')
ax_hist.spines[['top', 'right']].set_visible(False)
ax_hist.set_xlim()


# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
#ax1.set_title('Confidence Across Time')
ax.set_xlabel('Task trials')
ax.set_ylabel('Confidence\n(mean±SEM)')
#ax.legend(loc='upper left', bbox_to_anchor=(1,1))
ax.spines[['top', 'right']].set_visible(False)
#ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Task trials')
ax2.set_ylabel('Confidence')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
#ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
ax2.set_ylim(0, 100)

plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels for x axis
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels for y axis
plt.rc('legend', fontsize=20)    # fontsize of the legend


#plt.tight_layout()
path = r"C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil\metacognition-learning\comparative_models\results\variable_feedback\analysis"
save_path = os.path.join(path, 'confidence_over_time.png')
plt.savefig(save_path, dpi=300)
plt.show()

#%%
# Set df to df_a
df = df_a.copy()

# Initialize plot variables
colors = {'high': 'red', 'low': 'green'}

# Split data in high and low bdi
df['bdi_level'] = np.where(df.bdi.values < df.bdi.median(), 'low', 'high')


# Create subplots
fig, ((ax, ax0), (ax2, ax5))  = plt.subplots(2, 2,figsize=(20, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Add histogram to the right
divider = make_axes_locatable(ax2)
ax_hist_2 = divider.append_axes("right", size="20%", pad=0.4)

# Add histogram to the right
divider = make_axes_locatable(ax5)
ax_hist = divider.append_axes("right", size="20%", pad=0.4)

# Process data for each group
for bdi_level in ['high', 'low']:

    df_bdi = df[df.bdi_level == bdi_level]


    # Calculate mean and SEM for trial data
    trial_data = np.array([np.mean([participant[i]
                                    for participant
                                    in df_bdi['confidence_task']])
                           for i in range(15)])
    sem_trial = np.array([np.std([participant[i]
                                  for participant
                                  in df_bdi['confidence_task']],
                                 ddof=1) / np.sqrt(len(df))
                          for i in range(15)])


    # Combine the baseline and trial data
    combined_mean = trial_data
    combined_sem = sem_trial

    # Plot combined baseline and trial data as a continuous line
    x_combined = list(range(0, 15))
    ax.plot(x_combined, combined_mean, label=f'{bdi_level} BDI',
             color=colors[bdi_level], lw=2, marker='o')
    ax.fill_between(x_combined, combined_mean - combined_sem,
                     combined_mean + combined_sem, color=colors[bdi_level],
                     alpha=0.2)

    for p_trial in tqdm(df_bdi['confidence_task'],
                        total=len(df_bdi['confidence_task'])):

        trial_array = np.array(p_trial)
        p_data = trial_array # np.concatenate((baseline_array, trial_array))

        # plot
        derivative = np.gradient(p_data)
        ax2.plot(x_combined, p_data, label='',
                 color=colors[bdi_level], lw=0.3, alpha=1, marker='o',
                 markersize=0.9)



    # plot
    print(bdi_level)
    conf_flat = df_bdi.explode('confidence_task').confidence_task.values
    print(conf_flat[0])

    # Plot histogram on this new secondary axis
    ax_hist_2.hist(conf_flat, bins=25, orientation='horizontal',
                 color=colors[bdi_level], alpha=0.5)
    ax_hist_2.set_xlabel('Count')
    ax_hist_2.spines[['top', 'right']].set_visible(False)
    ax_hist_2.set_xlim()

    # Mean and sem for each participant
    means = [np.mean(participant)
             for participant in df_bdi['confidence_task']]
    sems = [np.std(participant, ddof=1) / np.sqrt(len(participant))
            for participant in df_bdi['confidence_task']]

    # Plotting error bars for each participant
    ax5.errorbar(range(len(means)), means, yerr=sems, fmt='o',
                color=colors[bdi_level])
    ax5.set_xlabel('Participants')
    ax5.set_ylabel('Confidence\n(mean+-SEM)')
    ax5.spines[['top', 'right']].set_visible(False)
    ax5.set_ylim(0, 100)

    # Plot histogram on this new secondary axis
    ax_hist.hist(means, bins=20, orientation='horizontal',
                 color=colors[bdi_level], alpha=0.3)

    ax_hist.set_xlabel('Count')
    ax_hist.spines[['top', 'right']].set_visible(False)
    ax_hist.set_xlim()
    ax_hist.set_ylim(0, 100)

# Hide ax0
ax0.set_visible(False)

# Configure the first subplot
#ax1.set_title('Confidence Across Time')
ax.set_xlabel('Task trials')
ax.set_ylabel('Confidence\n(mean±SEM)')
ax.legend(loc='upper left', bbox_to_anchor=(1,1))
ax.spines[['top', 'right']].set_visible(False)
#ax.axvline(0, ls='--', c='k', label='Start of Task Trials')

# Configure the other subplots
ax2.set_xlabel('Task trials')
ax2.set_ylabel('Confidence')
#ax2.legend()
ax2.spines[['top', 'right']].set_visible(False)
#ax2.axvline(0, ls='--', c='k', label='Start of Task Trials')
ax2.set_ylim(0, 100)

plt.rc('axes', titlesize=16)     # fontsize of the axes title
plt.rc('axes', labelsize=20)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels for x axis
plt.rc('ytick', labelsize=20)    # fontsize of the tick labels for y axis
plt.rc('legend', fontsize=20)    # fontsize of the legend


#plt.tight_layout()
path = r"C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil\metacognition-learning\comparative_models\results\variable_feedback\analysis"
save_path = os.path.join(path, 'confidence_over_time_bdi_split.png')
plt.savefig(save_path, dpi=300)
plt.show()

#%% Confidence variance not correlated with BDI
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Assuming df_a is already defined and df.confidence_task contains the data
df = df_a.copy()

# Calculate standard deviations for each confidence task
x = [np.std(vector) for vector in df.confidence_task]
y = df.bdi.values

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Prepare the scatter plot
fig, ax = plt.subplots(1,1)
plt.scatter(x, y, label='Data points')

# Prepare x values for the regression line plot
x_values = np.linspace(min(x), max(x), 100)

# Calculate y values for the regression line
y_values = intercept + slope * x_values

# Plot the regression line
plt.plot(x_values, y_values, 'r-', label=f'Regression line: y={intercept:.2f}+{slope:.2f}x')
#plt.legend()

# Annotate the plot with p-value and t-value
plt.annotate(f'p-value = {p_value:.3f}\nt-value = {slope/std_err:.2f}',
             xy=(0.05, 0.95), xycoords='axes fraction',
             verticalalignment='top',
             fontsize=10)

plt.xlabel('Standard Deviation of Confidence')
plt.ylabel('BDI Scores')
plt.title('Scatter Plot with Regression Line')
ax.spines[['top', 'right']].set_visible(False)

plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels for x axis
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels for y axis
plt.rc('legend', fontsize=12)    # fontsize of the legend


plt.show()

#%%



#%% learning to align confidence with feedback across time

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'high': 'red',
          'low': 'green',
          }
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharey=True)


for bdi_level in bdi_levels:
    # Filter data for current condition and BDI level
    df_cond = df_a[(df_a['bdi_level'] == bdi_level)]
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

    # Combine baseline and trial data for plotting
    combined_mean = trial_data #np.concatenate((baseline_data, trial_data))
    combined_sem = sem_trial #np.concatenate((baseline_sem, sem_trial))
    x_combined = list(range(len(trial_data)))

    # Plot combined baseline and trial data
    label = f'{bdi_level}'
    ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
    ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

# Set subplot title and labels
ax.set_ylabel('|Feedback avg - confidence|\n(mean±SEM)')
ax.set_xlabel('Trials')
ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
ax.spines[['top', 'right']].set_visible(False)
ax.legend()

plt.tight_layout()
plt.show()

#%% learning to align confidence with feedback across time
# Relative difference

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'high': 'red',
          'low': 'green',
          }
bdi_levels = ['high', 'low']

# Creating a subplot for each condition
fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharey=True)


for bdi_level in bdi_levels:
    # Filter data for current condition and BDI level
    df_cond = df_a[(df_a['bdi_level'] == bdi_level)]
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

    # Combine baseline and trial data for plotting
    combined_mean = trial_data#np.concatenate((baseline_data, trial_data))
    combined_sem = sem_trial#np.concatenate((baseline_sem, sem_trial))
    x_combined = list(range(len(trial_data)))

    # Plot combined baseline and trial data
    label = f'{bdi_level}'
    ax.plot(x_combined, combined_mean, label=label, color=colors[label], lw=2, marker='o')
    ax.fill_between(x_combined, combined_mean - combined_sem, combined_mean + combined_sem, color=colors[label], alpha=0.2)

# Set subplot title and labels
ax.set_ylabel('Feedback avg - confidence\n(mean±SEM)')
ax.set_xlabel('Trials')
ax.axhline(0, ls='--', c='k', alpha=0.5)

# Remove top right spines and add legend
ax.spines[['top', 'right']].set_visible(False)
ax.legend()

plt.tight_layout()
plt.show()





