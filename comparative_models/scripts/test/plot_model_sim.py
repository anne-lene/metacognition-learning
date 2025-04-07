# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 15:21:39 2024

@author: carll
"""

# Plot the average model max probabilities
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import ast
# Import data - fixed feedback condition
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_path = r"fixed_feedback"

#%% Load participant data

# Load DataFrame from JSON
load_path = os.path.join(project_path, 'Fixed_feedback_data.json')
df_a = pd.read_json(load_path, orient='records', lines=True)

# Debugging: Print a sample of the DataFrame
print(df_a.head())

#%%  Load comparative model simulation data

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_fit"
else:
    result_path = r"results\variable_feedback\trial_by_trial_fit"
filename = 'model_trial_data.csv'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, filename)
complete_df = pd.read_csv(filename)

#%% Get participant data plot

# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

# Set the font size globally
plt.rcParams.update({'font.size': 15})
plt.rcParams['xtick.labelsize'] = 15 # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = 15  # For y-axis tick labels

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

    x_trial = list(range(0, 20))
    x_combined =x_trial

    # Combine the baseline and trial data
    combined_mean = trial_data
    combined_sem = sem_trial

    # Plot combined baseline and trial data as a continuous line
    ax.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
            color=colors[cond], lw=2)
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

#ax.set_title('Mean Confidence Across Conditions')
ax.set_xlabel('Trials')
ax.set_ylabel('Confidence')
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Data')
plt.tight_layout()
plt.show()

#%% Plot the average model max probabilities
# Define the font size parameter
font_size = 12

# Update the default rc parameters for font size
plt.rcParams.update({'font.size': font_size})


def get_probability_dict(df):
    # Dictionary to store the results with keys as (pid, condition) and values as arrays of probabilities
    results = {}

    # Iterate over each combination of 'pid' and 'condition'
    for (pid, condition), group in df.groupby(['pid', 'condition']):

        # Sort the group by 'trial' and then 'option' to ensure consistent ordering
        group_sorted = group.sort_values(by=['trial', 'option'])

        # Find the unique number of trials and options to reshape the array accordingly
        n_trials = group_sorted['trial'].nunique()
        n_options = group_sorted['option'].nunique()

        # Reshape the probabilities into a 2D array where each row is a trial
        probabilities = group_sorted['probability'].values.reshape(n_trials, n_options)

        # Store the result in the dictionary
        results[(pid, condition)] = probabilities

    return results

def random_argmax(arr):
    # Step 1: Find the maximum value in the array
    max_value = np.max(arr)

    # Step 2: Find all indices where the array equals the maximum value
    max_indices = np.flatnonzero(arr == max_value)

    # Step 3: Randomly choose one index from the max indices
    chosen_index = np.random.choice(max_indices)

    return chosen_index

fig, [[ax,ax2],[ax3,ax4], [ax5,ax6],[ax7,ax8]] = plt.subplots(4,2,
                                                          figsize=(6,11),
                                                          sharex=True,
                                                          sharey=True)

# Check experiment (varied or fixed feedback)
if experiment_path != 'fixed_feedback':
    colors = ['blue', 'green', 'red', 'purple',
              'cyan', 'pink', 'orange', 'silver']
else:
    colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']

# Loop over models
for model, color, axis in tqdm(zip(complete_df.model.unique(), colors,
                             [ax, ax2, ax3, ax4, ax5, ax6, ax7, ax8]),
                         total=len(colors)):

    # Filter on current model
    df_model = complete_df[complete_df.model==model]

    # Get probability dictionary (contains one prob array per session)
    prob_dict = get_probability_dict(df_model)

    # Get the maximum probability option
    pid = []
    condition = []
    prob_max_idx_list = []
    # Loop over session keys: (session number, condition)
    np.random.seed(42)
    for key in prob_dict.keys():
        prob_array = prob_dict[key[0], key[1]] # Session array
        # If two or more values are highest, choose one of them at random
        prob_max_idx = []
        for row in prob_array:
            prob_max_idx.append(random_argmax(row))
        pid.append(key[0])
        condition.append(key[1])
        prob_max_idx_list.append(prob_max_idx)

    # Put prob_max_idx into dataframe
    d = {'pid': pid, 'condition': condition, 'prob_max_idx':prob_max_idx_list}
    df_model_max_idx = pd.DataFrame(d)

    # Make entries arrays
    df_model_max_idx['prob_max_idx'] = df_model_max_idx['prob_max_idx'].apply(np.array)


    # Plot
    for cond, color in zip(['neut', 'neg', 'pos'],
                           ['grey', 'red', 'green']):

        # Filter on condition
        df_plot = df_model_max_idx[df_model_max_idx.condition==cond]

        # Calculate mean and sem across columns
        means = np.mean(df_plot.prob_max_idx.values, axis=0)
        std = np.std(df_plot.prob_max_idx.values, axis=0, ddof=1)
        sem = std/np.sqrt(len(df_plot.prob_max_idx.values))

        # Plot
        x = range(1, len(means)+1)
        axis.plot(x, means, color=color)
        axis.fill_between(x, means+sem, means-sem,
                          color=color, alpha=0.2)


    axis.set_ylim(38,56)
    axis.set_xticks(range(1, len(means)+1, 2))
    axis.set_xticklabels(range(2, len(means)+2, 2))
    axis.set_xlabel('Trials')
    axis.set_ylabel('Confidence')
    if model == 'Delta_P_RW':
        axis.set_title('RW + Performance Delta')
    elif model == 'Choice_Kernel':
        axis.set_title('Choice Kernel')
    elif model == 'RW_Choice_Kernel':
        axis.set_title('RW + Choice Kernel')
    else:
        axis.set_title(model)

    axis.spines[['top', 'right']].set_visible(False)

    # Hide empty subplot
    if experiment_path == r"fixed_feedback":
        ax8.set_visible(False)
    else:
        pass

plt.tight_layout()
#plt.subplots_adjust(hspace=0.2)

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_fit"
else:
    result_path = r"results\variable_feedback\trial_by_trial_fit"

filename = 'model_max_prob_averge.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, filename)

# Save
#plt.savefig(save_path,
#            bbox_inches='tight',
#            dpi=300)

plt.show()


#%%
from matplotlib.patches import FancyBboxPatch

# Initialize variables for the first plot
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}
mean_performances = {cond: [] for cond in conditions}
sem_performances = {cond: [] for cond in conditions}
mean_baseline = {cond: [] for cond in conditions}
sem_baseline = {cond: [] for cond in conditions}

# Initialize the figure and subplots
fig, [[ax, ax2], [ax3, ax4], [ax5, ax6], [ax7, ax8]] = plt.subplots(4, 2, figsize=(13, 13), sharex=True, sharey=True)

# Set the font size globally
plt.rcParams.update({'font.size': 20})
plt.rcParams['xtick.labelsize'] = 28 # For x-axis tick labels
plt.rcParams['ytick.labelsize'] = 28  # For y-axis tick labels

# Subplot 1: Process trial and baseline data
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

    x_trial = list(range(0, 20))
    x_combined = x_trial

    # Combine the baseline and trial data
    combined_mean = trial_data
    combined_sem = sem_trial

    # Plot combined baseline and trial data as a continuous line
    ax.plot(x_combined, combined_mean, label=f'{cond.capitalize()} Condition',
            color=colors[cond], lw=2)
    ax.fill_between(x_combined,
                    [m - s for m, s in zip(combined_mean, combined_sem)],
                    [m + s for m, s in zip(combined_mean, combined_sem)],
                    color=colors[cond], alpha=0.2)

ax.set_xlabel('Trials', fontsize=26)
ax.set_ylabel('Confidence', fontsize=26)
ax.spines[['top', 'right']].set_visible(False)
ax.set_title('Data')


# Add a dark blue outline with rounded corners around the first subplot
bbox = FancyBboxPatch((-0.15, -0.15), 1.1, 1.3,
                      boxstyle="round,pad=0.1",
                      edgecolor='grey',
                      facecolor='grey',
                      linewidth=4,
                      alpha=0.1,
                      transform=ax.transAxes,
                      clip_on=False,
                      zorder=20)
ax.add_patch(bbox)

# Define the font size parameter for the other plots

def get_probability_dict(df):
    results = {}
    for (pid, condition), group in df.groupby(['pid', 'condition']):
        group_sorted = group.sort_values(by=['trial', 'option'])
        n_trials = group_sorted['trial'].nunique()
        n_options = group_sorted['option'].nunique()
        probabilities = group_sorted['probability'].values.reshape(n_trials, n_options)
        results[(pid, condition)] = probabilities
    return results

def random_argmax(arr):
    max_value = np.max(arr)
    max_indices = np.flatnonzero(arr == max_value)
    chosen_index = np.random.choice(max_indices)
    return chosen_index

# Check experiment (varied or fixed feedback)
experiment_path = 'fixed_feedback'  # or 'variable_feedback'
colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange', 'silver']

# Loop over models
for model, color, axis in tqdm(zip(complete_df.model.unique(), colors, [ax2, ax3, ax4, ax5, ax6, ax7, ax8]),
                               total=len(colors)):

    df_model = complete_df[complete_df.model == model]
    prob_dict = get_probability_dict(df_model)

    pid = []
    condition = []
    prob_max_idx_list = []
    np.random.seed(42)
    for key in prob_dict.keys():
        prob_array = prob_dict[key[0], key[1]]
        prob_max_idx = []
        for row in prob_array:
            prob_max_idx.append(random_argmax(row))
        pid.append(key[0])
        condition.append(key[1])
        prob_max_idx_list.append(prob_max_idx)

    d = {'pid': pid, 'condition': condition, 'prob_max_idx': prob_max_idx_list}
    df_model_max_idx = pd.DataFrame(d)
    df_model_max_idx['prob_max_idx'] = df_model_max_idx['prob_max_idx'].apply(np.array)

    for cond, color in zip(['neut', 'neg', 'pos'], ['grey', 'red', 'green']):
        df_plot = df_model_max_idx[df_model_max_idx.condition == cond]
        means = np.mean(df_plot.prob_max_idx.values, axis=0)
        std = np.std(df_plot.prob_max_idx.values, axis=0, ddof=1)
        sem = std / np.sqrt(len(df_plot.prob_max_idx.values))

        x = range(1, len(means) + 1)
        axis.plot(x, means, color=color)
        axis.fill_between(x, means + sem, means - sem, color=color, alpha=0.2)

    axis.set_ylim(38, 56)
    axis.set_xticks(range(1, len(means) + 1, 4))
    axis.set_xticklabels(range(2, len(means) + 2, 4))
    axis.set_xlabel('Trials', fontsize=26)
    axis.set_ylabel('Confidence', fontsize=26)
    if model == 'Delta_P_RW':
        axis.set_title('RW + Performance Delta')
    elif model == 'Choice_Kernel':
        axis.set_title('Choice Kernel')
    elif model == 'RW_Choice_Kernel':
        axis.set_title('RW + Choice Kernel')
    else:
        axis.set_title(model)

    axis.spines[['top', 'right']].set_visible(False)


plt.tight_layout()
plt.subplots_adjust(hspace=0.6)

save_path = os.path.join(project_path, 'comparative_models', result_path,
                         'model_max_prob_average.svg')
plt.savefig(save_path, bbox_inches='tight')

plt.show()

