# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:06:35 2024

@author: carll
"""

# Variance in dot estimate for the same number of dots
from src.utility_functions import (add_session_column)
import pandas as pd
from matplotlib import pyplot as plt
import os
from statsmodels.formula.api import mixedlm
from scipy.stats import zscore
#%% Import data
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

#%% Predict dot estimate stadnard deviation from trials

# Remove baseline trials
df_no_baseline = df[df.condition != 'baseline'].copy()

# Reset trial numbers to start at 0 and increment by 1 for each participant
df_no_baseline['trial'] = df_no_baseline.groupby(['pid', 'condition'])['trial'].transform(lambda x: pd.factorize(x)[0])

# Calculate standard deviation of 'estimate' over bins of 4 trials
df_no_baseline['trial_bin'] = df_no_baseline.groupby('pid')['trial'].transform(lambda x: (x // 4).astype(int))

# Group by 'pid', 'condition', 'trial_bin', and 'correct' and calculate the standard deviation of 'estimate'
variance_df = df_no_baseline.groupby(['pid',
                                      'condition',
                                      'trial_bin',
                                      'correct'])['estimate'].std().reset_index()

# Define the subplots grid
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 14))
axes = axes.flatten()

# Track the subplot index
plot_idx = 0
total_tests = 9  # 3 conditions * 3 correctness levels
cond_name = {'neut': 'Neutral',
             'pos': 'Positive',
             'neg': 'Negative'}

for cond in ['neut', 'pos', 'neg']:
    # Set up df
    condition_df = variance_df[variance_df.condition == cond]

    # Loop over correct
    for correct in set(df.correct):
        # Set up df
        correct_df = condition_df[condition_df.correct == correct].copy()

        # Calculate Z-scores for each participant's variance in estimates
        correct_df.loc[:, 'z_score'] = correct_df.groupby('pid')['estimate'].transform(lambda x: zscore(x, ddof=1))

        # Identify outliers (e.g., Z-score > 3 or < -3)
        outliers = correct_df[correct_df['z_score'].abs() > 3]['pid'].unique()

        # Remove outliers
        correct_df = correct_df[~correct_df['pid'].isin(outliers)]

        # Regression
        model = mixedlm.from_formula("estimate ~ trial_bin", correct_df,
                                     groups=correct_df["pid"],
                                     re_formula="1")
        result = model.fit()
        print(result.summary())

        # Calculate the Bonferroni corrected p-value
        p_value = result.pvalues['trial_bin']
        bonferroni_p_value = min(p_value * total_tests, 1.0)

        # Plot scatter and fit line on the current subplot
        ax = axes[plot_idx]
        ax.scatter(correct_df['trial_bin'], correct_df['estimate'],
                   alpha=0.5, label='Data')

        # Fit line
        x_vals = pd.Series(range(int(correct_df['trial_bin'].min()),
                                 int(correct_df['trial_bin'].max()) + 1))
        pred_df = pd.DataFrame({'trial_bin': x_vals})
        y_vals = result.predict(exog=pred_df)
        ax.plot(x_vals, y_vals, color='red', label='Fit Line')

        # Plot formatting
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Trial bin')
        ax.set_ylabel('Estimate Standard Deviation')
        ax.set_title(f'Condition: {cond_name[cond]}, Dots: {correct}')
        ax.legend(loc='center right')

        # Annotate the Bonferroni corrected p-value
        ax.annotate(f'Bonferroni corrected p-value: {bonferroni_p_value:.4f}',
                    xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                                           edgecolor='black',
                                           facecolor='white'))

        plot_idx += 1

plt.tight_layout()

plt.savefig('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/Std_dot_estimate_prediction_per_dot_number.png', dpi=300)
plt.show()

#%% Predicting Abs error from

# Remove baseline trials
df_no_baseline = df[df.condition != 'baseline'].copy()

# Reset trial numbers to start at 0 and increment by 1 for each participant
df_no_baseline['trial'] = df_no_baseline.groupby(['pid', 'condition'])['trial'].transform(lambda x: pd.factorize(x)[0])

# Calculate the mismatch between 'estimate' and 'correct'
df_no_baseline['mismatch'] = abs(df_no_baseline['estimate'] - df_no_baseline['correct'])

# Bin trials into groups of 4
df_no_baseline['trial_bin'] = df_no_baseline.groupby('pid')['trial'].transform(lambda x: (x // 4).astype(int))

# Group by 'pid', 'condition', 'trial_bin', and 'correct' and calculate the mean mismatch
mismatch_df = df_no_baseline.groupby(['pid', 'condition', 'trial', 'correct'])['mismatch'].mean().reset_index()

# Define the subplots grid
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 14))
axes = axes.flatten()

# Track the subplot index
plot_idx = 0
total_tests = 9  # 3 conditions * 3 correctness levels

for cond in ['neut', 'pos', 'neg']:
    # Set up df
    condition_df = mismatch_df[mismatch_df.condition == cond]

    # Loop over correct
    for correct in set(df.correct):
        # Set up df
        correct_df = condition_df[condition_df.correct == correct].copy()

        # Calculate Z-scores for each participant's mismatch
        correct_df.loc[:, 'z_score'] = correct_df.groupby('pid')['mismatch'].transform(lambda x: zscore(x, ddof=1))

        # Identify outliers (e.g., Z-score > 3 or < -3)
        outliers = correct_df[correct_df['z_score'].abs() > 3]['pid'].unique()

        # Remove outliers
        correct_df = correct_df[~correct_df['pid'].isin(outliers)]

        # Regression
        model = mixedlm("mismatch ~ trial", correct_df,
                        groups=correct_df["pid"], re_formula="1")
        result = model.fit()
        print(result.summary())

        # Calculate the Bonferroni corrected p-value
        p_value = result.pvalues['trial']
        bonferroni_p_value = min(p_value * total_tests, 1.0)

        # Plot scatter and fit line on the current subplot
        ax = axes[plot_idx]
        ax.scatter(correct_df['trial'], correct_df['mismatch'],
                   alpha=0.5, label='Data')

        # Fit line
        x_vals = pd.Series(range(int(correct_df['trial'].min()),
                                 int(correct_df['trial'].max()) + 1))
        pred_df = pd.DataFrame({'trial': x_vals})
        y_vals = result.predict(exog=pred_df)
        ax.plot(x_vals, y_vals, color='red', label='Fit Line')

        # Plot formatting
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlabel('Trial')
        ax.set_ylabel('Abs Error')
        ax.set_title(f'Condition: {cond}, Dots: {correct}')
        ax.legend(loc='center right')

        # Annotate the Bonferroni corrected p-value
        ax.annotate(f'Bonferroni corrected p-value: {bonferroni_p_value:.4f}',
                    xy=(0.5, 0.9), xycoords='axes fraction', ha='center',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3",
                                           edgecolor='black',
                                           facecolor='white'))

        plot_idx += 1

plt.tight_layout()

plt.savefig('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/Abs_Error_prediction_per_dot_number.png', dpi=300)
plt.show()
