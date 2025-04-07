# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:10:29 2024

@author: carll
"""

# Signs of asymetric learning rates


import numpy as np
from src.utility_functions import (add_session_column)
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf
from src.utility_functions import load_fixed_feedback_data


#%%
df_a = load_fixed_feedback_data()

#%% Restructure df

# Get bdi score for each participant
bdi_scores = df_a.groupby('pid')['bdi'].first().reset_index()

# Get BDI
df_a['bdi'] = df_a.groupby('pid')['bdi'].transform('first')

# Compute the median 'bdi' score
median_bdi = df_a['bdi'].median()

# Create a new column 'bdi_level' with 'high' or 'low' based on the median split
df_a['bdi_level'] = ['high' if x > median_bdi else 'low' for x in df_a['bdi']]

# Set up dataframe
df = df_a.copy()

# Create trial data where only single numbers
df['trial'] = [np.array(range(20)) for i in range(len(df))]
df['session'] = [[i]*20 for i in df['session'].values]
df['pid'] = [[i]*20 for i in df['pid'].values]
df['condition'] = [[i]*20 for i in df['condition_list'].values]
df['bdi'] = [[i]*20 for i in df['bdi'].values]
df['bdi_level'] = [[i]*20 for i in df['bdi_level'].values]

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
    'bdi': df['bdi'].values,
    'bdi_level': df['bdi_level'].values,
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
        'bdi': row['bdi'],
        'bdi_level': row['bdi_level'],
        #'feedback_sub_confidence': row['feedback_sub_confidence'],
    })

    # Append to the expanded DataFrame
    expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)

## Add bdi score of each participant
#expanded_df = pd.merge(expanded_df, bdi_scores, on='pid', how='left')

#%% Distribution of feedback prediction errors in each condition
df = expanded_df.copy()

# Initiate a figure
fig = plt.figure(figsize=(15,3))
# Loop over condition
for i, (cond, color) in enumerate(zip(df.condition.unique(),
                                      ['grey', 'red', 'green'])):

    # Set df to current condition
    df_cond = df[df.condition == cond]

    # Calcualte relative difference between feedback and confidence
    diff = df_cond.feedback - df_cond.confidence_task

    # Specify full condition name
    if cond == 'neut':
        condition = 'Neutral'
    if cond == 'neg':
        condition = 'Negative'
    if cond == 'pos':
        condition = 'Positive'

    ax = plt.subplot(1,3,i+1)
    plt.hist(diff, bins=20, color=color)
    plt.axvline(0, ls='-', c='k', label='0')
    ratio = round(diff[diff>=0].sum()/(diff[diff>=0].sum() + abs(diff[diff<0].sum())),
                  2)
    mean = round(np.mean(diff),2)
    plt.axvline(np.mean(diff), ls='--', c='k', label=f'mean={mean}')
    plt.title(f'{condition}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=(14), loc=('upper right'), bbox_to_anchor=(1.4,1))
    plt.xlabel('Prediction error')
    plt.ylabel('Count')

plt.subplots_adjust(wspace=0.7)
#plt.tight_layout()
plt.show()

print("""The conditions have a corresponding dissproportionate
      \namount of prediction errors""")


#%% Logistic regression - Feedback history explaining current confidence

# confidence_task = feedback-t1, feedback-t2, feedback-t3, feedback-t4...


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt

# Set up DataFrame
df = expanded_df.copy()



# Group by participant and calculate mean and standard deviation for 'confidence_task'
confidence_stats = df.groupby('pid')['confidence_task'].agg(['mean', 'std'])

# Join these statistics back to the original DataFrame
df = df.join(confidence_stats, on='pid', rsuffix='_conf_stats')

# Calculate Z-scores for 'confidence_task'
df['confidence_task_z'] = (df['confidence_task'] - df['mean']) / df['std']

# Apply the same Z-score transformation to 'feedback'
df['feedback_z'] = (df['feedback'] - df['mean']) / df['std']

# Adjust feedback to the transformed scale using mean and standard deviation of confidence_task
df['feedback_transformed'] = df['confidence_task_z'] * df['std'] + df['mean']

# Drop the extra columns if they are no longer needed
df.drop(['mean', 'std'], axis=1, inplace=True)


# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 5
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback_z'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

# Set up cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

error_participant_confidence = []
error_participant_feedback = []
error_pid = []

# Initiate figure
fig = plt.figure(figsize=(12,7))
for i, (cond, color) in enumerate(zip(df.condition.unique(),
                                      ['grey', 'red', 'green'])):

    # Set up condition dataframe
    df_cond = df[df.condition==cond]

    # Prepare to collect coefficients
    coefficients = []
    # Analyze each participant separately
    for pid in df_cond['pid'].unique():
        participant_data = df_cond[df_cond['pid'] == pid]
        X = participant_data[[f'feedback_t-{i}' for i in range(1, n_lag)]]
        y = participant_data['confidence_task_z']

        # Initialize and fit the model
        model = LinearRegression()
        model.fit(X, y)

        # Collect coefficients
        coefficients.append(model.coef_)

        if max(abs(model.coef_)) > 10 or y.var() < 1:
            error_participant_confidence.append(y)
            error_participant_feedback.append(X['feedback_t-1'])
            error_pid.append(pid)

    # Convert to numpy array for analysis
    coefficients = np.array(coefficients)

    # Calculate mean and SEM of coefficients
    coeff_mean = np.mean(coefficients, axis=0)
    coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

    # Specify full condition name
    if cond == 'neut':
        condition = 'Neutral'
    if cond == 'neg':
        condition = 'Negative'
    if cond == 'pos':
        condition = 'Positive'

    # Plotting the mean and SEM of the coefficients
    ax = plt.subplot(2,3,i+1)
    ax.plot(np.arange(1, n_lag), coeff_mean, color=color)

    plt.fill_between(np.arange(1, n_lag),
                     coeff_mean-coeff_sem,
                     coeff_mean+coeff_sem,
                     color=color,
                     alpha=0.5)
    plt.title(f'{condition}')
    plt.xlabel('Feedback')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
    ax.spines[['top', 'right']].set_visible(False)
    plt.ylim(-0.3, 0.3)
    ax2 = plt.subplot(2,3,i+4)

    for coeff in coefficients:
        ax2.plot(np.arange(1, n_lag), coeff, color=color)

    plt.title(f'{condition}')
    plt.xlabel('Feedback')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
    ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%% plot error participant
# =============================================================================
#
# fig = plt.figure(figsize=(20,4+(len(error_participant_feedback)*2)))
# for i in range(len(error_participant_feedback)):
#     ax = plt.subplot(len(error_participant_feedback),4,i+1)
#     plt.scatter(y=error_participant_feedback[i], x=error_participant_confidence[i],
#                 label=f'PID: {error_pid[0]}')
#     plt.ylabel('previous feedback')
#     plt.xlabel('current confidence')
#     ax.spines[['top', 'right']].set_visible(False)
# #plt.legend()
# plt.tight_layout()
# plt.show()
#
# print("""Takeaway:
#       \nParticipant only chooses the same confidence except for once,
#       \nwhich makes the regression hard to fit.""")
# =============================================================================

#%% Remove participant and plot again

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 5
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

# Set up cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

error_participant_confidence = []
error_participant_feedback = []
# Initiate figure
fig = plt.figure(figsize=(12,7))
for i, (cond, color) in enumerate(zip(df.condition.unique(),
                                      ['grey', 'red', 'green'])):

    # Set up condition dataframe
    df_cond = df[df.condition==cond]

    # Prepare to collect coefficients
    coefficients = []
    # Analyze each participant separately
    for pid in df_cond['pid'].unique():
        participant_data = df_cond[df_cond['pid'] == pid]
        X = participant_data[[f'feedback_t-{i}' for i in range(1, n_lag)]]
        y = participant_data['confidence_task']

        # Initialize and fit the model
        model = LinearRegression()
        model.fit(X, y)

        if max(abs(model.coef_)) > 10:
            error_participant_confidence.append(y)
            error_participant_feedback.append(X['feedback_t-1'])
        else:
            # Collect coefficients
            coefficients.append(model.coef_)

    # Convert to numpy array for analysis
    coefficients = np.array(coefficients)

    # Calculate mean and SEM of coefficients
    coeff_mean = np.mean(coefficients, axis=0)
    coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

    # Specify full condition name
    if cond == 'neut':
        condition = 'Neutral'
    if cond == 'neg':
        condition = 'Negative'
    if cond == 'pos':
        condition = 'Positive'

    # Plotting the mean and SEM of the coefficients
    ax = plt.subplot(2,3,i+1)
    ax.plot(np.arange(1, n_lag), coeff_mean, color=color)

    plt.fill_between(np.arange(1, n_lag),
                     coeff_mean-coeff_sem,
                     coeff_mean+coeff_sem,
                     color=color,
                     alpha=0.5)
    plt.title(f'{condition}')
    plt.xlabel('Feedback')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
    ax.spines[['top', 'right']].set_visible(False)
    plt.ylim(-0.2, 0.3)

    ax2 = plt.subplot(2,3,i+4)
    for coeff in coefficients:
        ax2.plot(np.arange(1, n_lag), coeff, color=color)

    plt.title(f'{condition}')
    plt.xlabel('Feedback')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
    ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

#%% single plot

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 3
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

var_list = []
error_participant_confidence = []
error_participant_feedback = []
# Initiate figure
fig = plt.figure(figsize=(4,6))
for i, (cond, color) in enumerate(zip(df.condition.unique(),
                                      ['grey', 'red', 'green'])):

    # Set up condition dataframe
    df_cond = df[df.condition==cond]

    coefficients = []

    # Analyze each participant separately
    for pid in df_cond['pid'].unique():
        participant_data = df_cond[df_cond['pid'] == pid]
        X = participant_data[[f'feedback_t-{i}'
                              for i
                              in range(1, n_lag)]]
        y = participant_data['confidence_task']

        # Add constant to the predictor variables for statsmodels
        X = sm.add_constant(X)

        # Initialize and fit the model
        model = sm.OLS(y, X).fit()

        # Get variance of the confidence task
        variance = participant_data['confidence_task'].var()
        var_list.append(variance)

        # Check if any p-value is greater than the significance level,
        # e.g., 0.05, or other conditions
        if np.any(model.pvalues['feedback_t-1'] > 1): #or max(abs(model.params[1:])) > 10 or variance < 2:
            error_participant_confidence.append(y)
            error_participant_feedback.append(X.iloc[:, 1:])  # Exclude the constant column
        else:
            # Collect coefficients, excluding the intercept
            a = 0

            coefficients.append([model.params[f'feedback_t-{i}'] for i in range(1, n_lag)])

    # Convert to numpy array for analysis
    coefficients = np.array(coefficients)

    # Exclude outliers
    # Calculate Q1 and Q3
    data = coefficients
    Q1 = np.percentile(data, 25, axis=0)
    Q3 = np.percentile(data, 75, axis=0)

    # Calculate the IQR
    IQR = Q3 - Q1

    # Determine outliers and clean data
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    outliers = (data < lower_fence) | (data > upper_fence)

    # Create a mask for non-outliers
    non_outliers = ~outliers.any(axis=1)

    # Filter data to keep only non-outliers
    coefficients = data[non_outliers]

    # Calculate mean and SEM of coefficients
    coeff_mean = np.mean(coefficients, axis=0)
    coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

    # Specify full condition name
    if cond == 'neut':
        condition = 'Neutral'
    if cond == 'neg':
        condition = 'Negative'
    if cond == 'pos':
        condition = 'Positive'

    # Plotting the mean and SEM of the coefficients
    ax = plt.subplot(2,1,1)
    ax.plot(np.arange(1, n_lag), coeff_mean, color=color,
            label=f'{condition}')

    plt.fill_between(np.arange(1, n_lag),
                     coeff_mean-coeff_sem,
                     coeff_mean+coeff_sem,
                     color=color,
                     alpha=0.5)

    plt.legend(fontsize=(10))
    plt.xlabel('Feedback')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
    ax.spines[['top', 'right']].set_visible(False)

    ax2 = plt.subplot(2,1,2)

    for coeff in coefficients:
        ax2.plot(np.arange(1, n_lag), coeff, color=color, lw=0.5)

    #plt.title(f'{condition}')
    plt.xlabel('Feedback')
    plt.ylabel('Coefficients')
    plt.xticks(rotation=45)
    plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
    ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%% plot overall varaince in confidence across participants

plt.hist(var_list)


#%% Split into low vs high bdi

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 2
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

error_participant_confidence = []
error_participant_feedback = []
all_coefficients = []
pids = []

# Initiate figures
fig = plt.figure(figsize=(8,6))

for bdi_nr, bdi_level in enumerate(df.bdi_level.unique()):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]

    for i, (cond, color) in enumerate(zip(df_bdi.condition.unique(),
                                          ['grey', 'red', 'green'])):

        # Set up condition dataframe
        df_cond = df_bdi[df_bdi.condition==cond]

        # Prepare to collect coefficients
        coefficients = []

        # Analyze each participant separately
        for pid in df_cond['pid'].unique():
            participant_data = df_cond[df_cond['pid'] == pid]
            X = participant_data[[f'feedback_t-{i}' for i in range(1, n_lag)]]
            y = participant_data['confidence_task']

            # Add constant to the predictor variables for statsmodels
            X = sm.add_constant(X)

            # Initialize and fit the model
            model = sm.OLS(y, X).fit()

            # Get variance of the confidence task
            variance = participant_data['confidence_task'].var()
            var_list.append(variance)

            # Check if any p-value is greater than the significance level, e.g., 0.05, or other conditions
            if np.any(model.pvalues[['feedback_t-1',]] > 1): #or max(abs(model.params[1:])) > 10 or variance < 2:
                error_participant_confidence.append(y)
                error_participant_feedback.append(X.iloc[:, 1:])  # Exclude the constant column
            else:
                # Collect coefficients, excluding the intercept
                a = 0

            coefficients.append(model.params[['feedback_t-1', 'feedback_t-2']].tolist())

        # Convert to numpy array for analysis
        coefficients = np.array(coefficients)
        all_coefficients.append(coefficients)
        pids.append(pid)

        # Exclude outliers
        # Calculate Q1 and Q3
        data = coefficients
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)

        # Calculate the IQR
        IQR = Q3 - Q1

        # Determine outliers and clean data
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        outliers = (data < lower_fence) | (data > upper_fence)

        # Create a mask for non-outliers
        non_outliers = ~outliers.any(axis=1)

        # Filter data to keep only non-outliers
        coefficients = data[non_outliers]

        # Calculate mean and SEM of coefficients
        coeff_mean = np.mean(coefficients, axis=0)
        coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

        # Specify full condition name
        if cond == 'neut' and bdi_level=='low':
            condition = 'Neutral'
            color = 'darkgrey'
            plot = 1
        if cond == 'neut' and bdi_level=='high':
            condition = 'Neutral'
            color ='black'
            plot = 2

        if cond == 'neg' and bdi_level=='low':
            condition = 'Negative'
            color = 'red'
            plot = 1
        if cond == 'neg' and bdi_level=='high':
            condition = 'Negative'
            color = 'darkred'
            plot = 2

        if cond == 'pos' and bdi_level=='low':
            condition = 'Positive'
            color = 'lightgreen'
            plot = 1
        if cond == 'pos' and bdi_level=='high':
            condition = 'Positive'
            color = 'darkgreen'
            plot = 2

        # Plotting the mean and SEM of the coefficients
        ax = plt.subplot(2,2,plot)
        ax.plot(np.arange(1, n_lag), coeff_mean, color=color,
                label=f'{condition}, {bdi_level} BDI')

        ax.errorbar(x=np.arange(1, n_lag), y=coeff_mean, yerr=coeff_sem,
                    fmt='o', capsize=5, color=color)

        plt.fill_between(np.arange(1, n_lag),
                         coeff_mean-coeff_sem,
                         coeff_mean+coeff_sem,
                         color=color,
                         alpha=0.5)

        plt.legend(fontsize=(10))
        plt.xlabel('Feedback', fontsize=15)
        plt.ylabel('Coefficients', fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(rotation=45, fontsize=15)
        plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
        ax.spines[['top', 'right']].set_visible(False)
        plt.ylim(-0.05, 0.3)
        ax2 = plt.subplot(2,2,plot+2)

        for coeff in coefficients:
            ax2.plot(np.arange(1, n_lag), coeff, color=color, lw=1)

        #plt.title(f'{condition}')
        plt.xlabel('Feedback',  fontsize=15)
        plt.ylabel('Coefficients',  fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(rotation=45, fontsize=15)
        plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
        ax2.spines[['top', 'right']].set_visible(False)
        plt.ylim(-0.5, 1)

plt.tight_layout()
plt.show()

#%% remove non sig coeffs - low vs high bdi

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 1
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

error_participant_confidence = []
error_participant_feedback = []
all_coefficients = []
pids = []

# Initiate figures
fig = plt.figure(figsize=(8,6))

for bdi_nr, bdi_level in enumerate(df.bdi_level.unique()):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]

    for i, (cond, color) in enumerate(zip(['neut', 'pos', 'neg'],
                                          ['grey', 'green', 'red'])):

        # Set up condition dataframe
        df_cond = df_bdi[df_bdi.condition==cond]

        # Prepare to collect coefficients
        coefficients = []

        # Analyze each participant separately
        for pid in df_cond['pid'].unique():
            participant_data = df_cond[df_cond['pid'] == pid]
            X = participant_data[[f'feedback_t-{i}' for i in range(1, n_lag)]]
            y = participant_data['confidence_task']

            # Add constant to the predictor variables for statsmodels
            X = sm.add_constant(X)

            # Fit the OLS model
            model = sm.OLS(y, X).fit()

            # Check if any p-value is greater than the significance level, e.g., 0.05
            if np.any(model.pvalues['feedback_t-1'] > 0.05):# or max(abs(model.params[1:])) > 10 or y.var() < 1:
                error_participant_confidence.append(y)
                error_participant_feedback.append(X['feedback_t-1'])
            else:
                # Collect coefficients
                coefficients.append(model.params['feedback_t-1'])  # Exclude the intercept

        # Convert to numpy array for analysis
        coefficients = np.array(coefficients)
        all_coefficients.append(coefficients)
        pids.append(pid)

# =============================================================================
#         # Exclude outliers
#         # Calculate Q1 and Q3
#         data = coefficients
#         Q1 = np.percentile(data, 25, axis=0)
#         Q3 = np.percentile(data, 75, axis=0)
#
#         # Calculate the IQR
#         IQR = Q3 - Q1
#
#         # Determine outliers and clean data
#         lower_fence = Q1 - 1.5 * IQR
#         upper_fence = Q3 + 1.5 * IQR
#         outliers = (data < lower_fence) | (data > upper_fence)
#
#         # Create a mask for non-outliers
#         non_outliers = ~outliers.any()
#
#         # Filter data to keep only non-outliers
#         coefficients = data[non_outliers]
# =============================================================================

        # count coeffs
        prop_sig = len(coefficients) / len(df_cond.pid.unique())
        print(f'proportion sig previous feedback coeffs: {cond}, {round(prop_sig*100, 2)}%' )

        # Calculate mean and SEM of coefficients
        coeff_mean = np.mean(coefficients, axis=0)
        coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

        # Specify full condition name
        if cond == 'neut' and bdi_level=='low':
            condition = 'Neutral'
            color = 'darkgrey'
            plot = 1
        if cond == 'neut' and bdi_level=='high':
            condition = 'Neutral'
            color ='black'
            plot = 2

        if cond == 'neg' and bdi_level=='low':
            condition = 'Negative'
            color = 'red'
            plot = 1
        if cond == 'neg' and bdi_level=='high':
            condition = 'Negative'
            color = 'darkred'
            plot = 2

        if cond == 'pos' and bdi_level=='low':
            condition = 'Positive'
            color = 'lightgreen'
            plot = 1
        if cond == 'pos' and bdi_level=='high':
            condition = 'Positive'
            color = 'darkgreen'
            plot = 2

        # Plotting the mean and SEM of the coefficients
        ax = plt.subplot(2,2,plot)
        ax.plot(np.arange(1, n_lag), coeff_mean, color=color,
                label=f'{condition}, {bdi_level} BDI')

        ax.errorbar(x=np.arange(1, n_lag), y=coeff_mean, yerr=coeff_sem,
                    fmt='o', capsize=5, color=color)

        plt.fill_between(np.arange(1, n_lag),
                         coeff_mean-coeff_sem,
                         coeff_mean+coeff_sem,
                         color=color,
                         alpha=0.5)

        plt.legend(fontsize=(10))
        plt.xlabel('Feedback')
        plt.ylabel('Coefficients')
        plt.xticks(rotation=45)
        plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
        ax.spines[['top', 'right']].set_visible(False)
        #plt.ylim(-0.05, 0.2)
        ax2 = plt.subplot(2,2,plot+2)

        for coeff in coefficients:
            ax2.scatter(np.arange(1, n_lag), coeff, color=color, lw=1)

        #plt.title(f'{condition}')
        plt.xlabel('Feedback')
        plt.ylabel('Coefficients')
        plt.xticks(rotation=45)
        plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
        ax2.spines[['top', 'right']].set_visible(False)
        #plt.ylim(-0.5, 1)

plt.tight_layout()
plt.show()


#%% coeff assymetry - low vs high bdi

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 1
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

# Filter the DataFrame to only include 'neg' and 'pos' conditions
filtered_df = df[df['condition'].isin(['neg', 'pos'])]

# Group by 'pid' and collect all unique conditions per participant
grouped = filtered_df.groupby('pid')['condition'].unique()

# Filter groups to find PIDs with both 'neg' and 'pos'
pids_with_both_conditions = grouped[grouped.apply(lambda x: 'neg' in x and 'pos' in x)].index.tolist()

# Filter the DataFrame to only include participants with both 'neg' and 'pos' conditions
condition_filtered_df = df[df['pid'].isin(pids_with_both_conditions)]

# Split the filtered DataFrame based on the 'bdi_level' flag
high_bdi_pids = condition_filtered_df[condition_filtered_df['bdi_level'] == 'high']['pid'].tolist()
low_bdi_pids = condition_filtered_df[condition_filtered_df['bdi_level'] == 'low']['pid'].tolist()



# Prepare to collect coefficients
coefficients = []
pids = []
sig = []
p_value = []
bdi_levels = []
cond_list = []


# Initiate figures
fig = plt.figure(figsize=(8,6))

for bdi_nr, bdi_level in enumerate(df.bdi_level.unique()):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]

    for i, (cond, color) in enumerate(zip(['pos',   'neg'],
                                          ['green', 'red'])):

        # Set up condition dataframe
        df_cond = df_bdi[df_bdi.condition==cond]


        # Analyze each participant separately
        if bdi_level == 'high':
            list_of_pids = high_bdi_pids
        if bdi_level == 'low':
            list_of_pids = low_bdi_pids

        for pid in set(list_of_pids):
            participant_data = df_cond[df_cond['pid'] == pid]
            X = participant_data[[f'feedback_t-{i}' for i in range(1, n_lag)]]
            y = participant_data['confidence_task']

            # Add constant to the predictor variables for statsmodels
            X = sm.add_constant(X)

            # Fit the OLS model
            model = sm.OLS(y, X).fit()

            # Check if any p-value is greater than the significance level, e.g., 0.05
            if np.any(model.pvalues['feedback_t-1'] > 0.05):# or max(abs(model.params[1:])) > 10 or y.var() < 1:
                sig.append(False)
            else:
                sig.append(True)

            p_value.append(model.pvalues['feedback_t-1'])
            coefficients.append(model.params['feedback_t-1'])
            bdi_levels.append(bdi_level)
            cond_list.append(cond)
            pids.append(pid)

# Create df
d = {'pids': pids,
     'coefficients': coefficients,
     'p_value': p_value,
     'sig': sig,
     'bdi_levels': bdi_levels,
     'condition': cond_list,
     }

df_plot = pd.DataFrame(d)
#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Assuming df_plot is already created and contains the appropriate columns
# Filter out only significant coefficients
df_sig = df_plot#[df_plot['sig'] == True]

# Calculate asymmetry for each participant between conditions 'pos' and 'neg'
asymmetry = df_sig.pivot_table(index=['pids', 'bdi_levels'],
                               columns='condition', values='coefficients')
asymmetry['asymmetry'] = abs(asymmetry['pos']) / (abs(asymmetry['pos']) + abs(asymmetry['neg']))

# Reset index for grouping
asymmetry.reset_index(inplace=True)

# Group by BDI levels to calculate mean and SEM
stats_df = asymmetry.groupby('bdi_levels')['asymmetry'].agg(['mean', 'sem'])

# Plotting
fig, ax = plt.subplots(1,1, figsize=(6,4))
colors = {'low': 'blue', 'high': 'red'}

# Plot individual asymmetry values

for label, group in asymmetry.groupby('bdi_levels'):
    ax.scatter([label]*len(group), group['asymmetry'], color=colors[label], alpha=0.3)
    print(label)

# Perform the independent samples t-test
cleaned_asymmetry = asymmetry.dropna(subset=['asymmetry'])
group1, group2 = cleaned_asymmetry.groupby('bdi_levels')['asymmetry']
means = asymmetry.groupby('bdi_levels')['asymmetry'].mean()
sems = asymmetry.groupby('bdi_levels')['asymmetry'].sem()
t_stat, p_value = stats.ttest_ind(group1[1], group2[1])
x_coords = [0, 1]
y_max = max(means) + max(sems)
ax.annotate(f'T={t_stat:.2f}, p={p_value:.3f}',
            xy=((x_coords[0]+x_coords[1])/2, y_max),
            textcoords='data',
            ha='center', va='top', fontsize=10,
            arrowprops=dict(arrowstyle='-', lw=0.5),
            )

# Plot mean ± SEM for each BDI level
for level, row in stats_df.iterrows():
    mean = row['mean']
    sem = row['sem']
    ax.errorbar(level, mean, yerr=sem, fmt='o', color=colors[level], label=f'{level} BDI')

ax.set_xlabel('BDI Level')
ax.set_ylabel('Asymmetry of Coefficient')
ax.spines[['top', 'right']].set_visible(False)
#ax.set_title('Asymmetry Coefficient Distribution by BDI Level')
ax.legend()

plt.show()

#%%
# Calculate assymetry for low bdi
assym = all_coefficients_low[0] / (all_coefficients_low[0] + all_coefficients_low[1])

# Calculate mean and SEM of coefficients
coeff_mean = np.mean(coefficients, axis=0)
coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

# Specify full condition name
if cond == 'neut' and bdi_level=='low':
    condition = 'Neutral'
    color = 'darkgrey'
    plot = 1
if cond == 'neut' and bdi_level=='high':
    condition = 'Neutral'
    color ='black'
    plot = 2

if cond == 'neg' and bdi_level=='low':
    condition = 'Negative'
    color = 'red'
    plot = 1
if cond == 'neg' and bdi_level=='high':
    condition = 'Negative'
    color = 'darkred'
    plot = 2

if cond == 'pos' and bdi_level=='low':
    condition = 'Positive'
    color = 'lightgreen'
    plot = 1
if cond == 'pos' and bdi_level=='high':
    condition = 'Positive'
    color = 'darkgreen'
    plot = 2

# Plotting the mean and SEM of the coefficients
ax = plt.subplot(2,2,plot)
ax.plot(np.arange(1, n_lag), coeff_mean, color=color,
        label=f'{condition}, {bdi_level} BDI')

ax.errorbar(x=np.arange(1, n_lag), y=coeff_mean, yerr=coeff_sem,
            fmt='o', capsize=5, color=color)

plt.fill_between(np.arange(1, n_lag),
                 coeff_mean-coeff_sem,
                 coeff_mean+coeff_sem,
                 color=color,
                 alpha=0.5)

plt.legend(fontsize=(10))
plt.xlabel('Feedback')
plt.ylabel('Coefficients')
plt.xticks(rotation=45)
plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
ax.spines[['top', 'right']].set_visible(False)
#plt.ylim(-0.05, 0.2)
ax2 = plt.subplot(2,2,plot+2)

for coeff in coefficients:
    ax2.scatter(np.arange(1, n_lag), coeff, color=color, lw=1)

#plt.title(f'{condition}')
plt.xlabel('Feedback')
plt.ylabel('Coefficients')
plt.xticks(rotation=45)
plt.xticks(np.arange(1, n_lag), [f't-{i}' for i in range(1, n_lag)])
ax2.spines[['top', 'right']].set_visible(False)
#plt.ylim(-0.5, 1)

plt.tight_layout()
plt.show()

#%% Split into low vs high bdi but only t-1
import statsmodels.api as sm

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 1
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

# Initiate result df
results_df = pd.DataFrame(columns=['pid', 'condition', 'bdi_level', 'coefficients', 'prev_feedback_is_sig'])

data_rows = []
for bdi_nr, bdi_level in enumerate(df.bdi_level.unique()):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]

    for i, (cond, color) in enumerate(zip(df_bdi.condition.unique(),
                                          ['grey', 'red', 'green'])):

        # Set up condition dataframe
        df_cond = df_bdi[df_bdi.condition==cond]

        # Prepare to collect coefficients
        coefficients = []
        # Analyze each participant separately
        for pid in df_cond['pid'].unique():
            participant_data = df_cond[df_cond['pid'] == pid]
            X = participant_data[[f'feedback_t-1' for i in range(1, n_lag)]]
            y = participant_data['confidence_task']

            # Add a constant to the model
            # (statsmodels does not add it by default)
            X = sm.add_constant(X)

            # Initialize and fit the OLS model
            model = sm.OLS(y, X).fit()

            # Check if any coefficient is significant
            if any(model.pvalues[1:] < 0.05):  # We skip the constant's p-value
                #print(f'sig {model.params[1:].values}')
                #coefficients.append(model.params[1])  # Collect coefficients, excluding constant
                new_row = data_rows.append({
                        'pid': pid,
                        'condition': cond,
                        'bdi_level': bdi_level,
                        'coefficients': model.params['feedback_t-1'],
                        'prev_feedback_is_sig': True,
                    })
            else:
                new_row = data_rows.append({
                        'pid': pid,
                        'condition': cond,
                        'bdi_level': bdi_level,
                        'coefficients':model.params['feedback_t-1'],
                        'prev_feedback_is_sig': False,
                    })

results_df = pd.DataFrame(data_rows)




#%%

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plot settings
plt.figure(figsize=(7, 3))
plt.rcParams['axes.titlesize'] = 16  # Title size
plt.rcParams['axes.labelsize'] = 16  # X and Y labels size
plt.rcParams['xtick.labelsize'] = 14  # X tick labels size
plt.rcParams['ytick.labelsize'] = 14  # Y tick labels size
plt.rcParams['legend.fontsize'] = 16  # Legend font size
plt.rcParams['font.size'] = 14  # Text size


for bdi_nr, bdi_level in enumerate(df.bdi_level.unique()):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]

    for i, (cond, color) in enumerate(zip(df_bdi.condition.unique(),
                                          ['grey', 'red', 'green'])):
        # Subset for condition and BDI level
        subset = results_df[(results_df['condition'] == cond) & (results_df['bdi_level'] == bdi_level)]
        print(len(subset.pid.unique()))
        # Calculate mean and SEM
        coeffs = subset[subset['prev_feedback_is_sig']==True]['coefficients']
        #coeffs = subset['coefficients']

        # Exclude outliers
        # Calculate Q1 and Q3
        data = coeffs
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)

        # Calculate the IQR
        IQR = Q3 - Q1

        # Determine outliers and clean data
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        outliers = (data < lower_fence) | (data > upper_fence)

        # Create a mask for non-outliers
        non_outliers = ~outliers

        # Filter data to keep only non-outliers
        coeffs = data[non_outliers]


        mean_coeffs = coeffs.mean()
        sem_coeffs = coeffs.sem()

        # Specify full condition name
        if cond == 'neut' and bdi_level=='low':
            condition = 'Neutral'
            color = 'darkgrey'
            plot = 1
        if cond == 'neut' and bdi_level=='high':
            condition = 'Neutral'
            color ='black'
            plot = 2

        if cond == 'neg' and bdi_level=='low':
            condition = 'Negative'
            color = 'red'
            plot = 1
        if cond == 'neg' and bdi_level=='high':
            condition = 'Negative'
            color = 'darkred'
            plot = 2

        if cond == 'pos' and bdi_level=='low':
            condition = 'Positive'
            color = 'lightgreen'
            plot = 1
        if cond == 'pos' and bdi_level=='high':
            condition = 'Positive'
            color = 'darkgreen'
            plot = 2

        # Plotting the mean and SEM of the coefficients
        ax = plt.subplot(1,2,plot)

        ax.errorbar(x=i,
                    y=mean_coeffs,
                    yerr=sem_coeffs,
                    fmt='o',
                    capsize=5,
                    color=color,
                    label=f'{condition}, {bdi_level}')

        plt.legend(fontsize=(9),
                   loc='best',
                   #bbox_to_anchor=(0.1,1),
                   )
        plt.xlabel('Condition')
        plt.ylabel('Coefficients')
        plt.xticks(rotation=45)
        plt.xticks(range(0,3), [f'{c}' for c
                                     in ['', '', '']])

        ax.scatter(np.linspace(i-.25, i+.25, len(coeffs)),
                   coeffs,
                   color=color,
                   s=3,
                   alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        plt.ylim(-0.5, 1.5)
        plt.xlim(-1,3)


plt.tight_layout()
plt.show()








#%%
                # BDIs.append(participant_data['bdi'].iloc[0])  # Assuming 'bdi' is not unique per pid

# =============================================================================
#             if y.var()<1:
#                 error_participant_confidence.append(y)
#                 error_participant_feedback.append(X['feedback_t-1'])
#             else:
#                 # Collect coefficients
#                 coefficients.append(model.coef_)
# =============================================================================

        # Convert to numpy array for analysis
        coefficients = np.array(coefficients)
        all_coefficients.append(coefficients)
        pids.append(pid)

        # Exclude outliers
        # Calculate Q1 and Q3
        data = coefficients
        Q1 = np.percentile(data, 25, axis=0)
        Q3 = np.percentile(data, 75, axis=0)

        # Calculate the IQR
        IQR = Q3 - Q1

        # Determine outliers and clean data
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        outliers = (data < lower_fence) | (data > upper_fence)

        # Create a mask for non-outliers
        non_outliers = ~outliers.any(axis=1)

        # Filter data to keep only non-outliers
        coefficients = data[non_outliers]

        # Calculate mean and SEM of coefficients
        coeff_mean = np.mean(coefficients, axis=0)
        coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

        # Specify full condition name
        if cond == 'neut' and bdi_level=='low':
            condition = 'Neutral'
            color = 'darkgrey'
            plot = 1
        if cond == 'neut' and bdi_level=='high':
            condition = 'Neutral'
            color ='black'
            plot = 2

        if cond == 'neg' and bdi_level=='low':
            condition = 'Negative'
            color = 'red'
            plot = 1
        if cond == 'neg' and bdi_level=='high':
            condition = 'Negative'
            color = 'darkred'
            plot = 2

        if cond == 'pos' and bdi_level=='low':
            condition = 'Positive'
            color = 'lightgreen'
            plot = 1
        if cond == 'pos' and bdi_level=='high':
            condition = 'Positive'
            color = 'darkgreen'
            plot = 2

        # Plotting the mean and SEM of the coefficients
        ax = plt.subplot(1,2,plot)

        ax.errorbar(x=i,
                    y=coeff_mean,
                    yerr=coeff_sem,
                    fmt='o',
                    capsize=5,
                    color=color,
                    label=f'{condition}, {bdi_level}')

        plt.legend(fontsize=(8),
                   loc='best',
                   #bbox_to_anchor=(0.1,1),
                   )
        plt.xlabel('Condition')
        plt.ylabel('Coefficients')
        plt.xticks(rotation=45)
        plt.xticks(range(0,3), [f'{c}'
                                     for c
                                     in ['Neutral', 'Negative', 'Positive']])

        ax.scatter(np.linspace(i-.25, i+.25, len(coefficients)),
                   coefficients,
                   color=color,
                   s=3,
                   alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

        plt.ylim(-0.5, 1.5)
        plt.xlim(-1,3)


plt.tight_layout()
plt.show()

#%% Identify outliers

all_coefficients_array = np.concatenate(all_coefficients)
all_pids_array = np.array(pids)

data = all_coefficients_array

# Calculate Q1 and Q3
Q1 = np.percentile(data, 25, axis=0)
Q3 = np.percentile(data, 75, axis=0)

# Calculate the IQR
IQR = Q3 - Q1

# Determine outliers
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = np.where((data < lower_fence) | (data > upper_fence))

print("Outliers found at indices:", list(zip(outliers[0], outliers[1])))
print("Outlier values:", data[outliers])


#%% Split into low vs high bdi

# Set up DataFrame
df = expanded_df.copy()

# Creating lagged feedback columns for 1 to 10 trials back
n_lag = 1
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'feedback_t-{i}'] = df.groupby(['pid', 'session'])['feedback'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

# Set up cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

error_participant_confidence = []
error_participant_feedback = []

optimisms = []
BDIs = []
PIDs = []
BDI_level = []

# Initiate figure
fig = plt.figure(figsize=(10,6))

for bdi_nr, bdi_level in enumerate(df.bdi_level.unique()):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]


    # Analyze each participant separately
    for pid in df_bdi['pid'].unique():

        participant_data = df_bdi[df_bdi['pid'] == pid]

        # Prepare to collect coefficients
        pos_coefficient = []
        neg_coefficient = []
        for i, (cond, color) in enumerate(
                                       zip(participant_data.condition.unique(),
                                           ['grey', 'red', 'green'])):

            # Set up condition dataframe
            df_cond = participant_data[participant_data.condition==cond]

            X = df_cond[[f'feedback_t-{i}' for i in range(1, n_lag)]]
            y = df_cond['confidence_task']

            # Initialize and fit the model
            model = LinearRegression()
            model.fit(X, y)


            if max(abs(model.coef_)) > 10 or y.var() < 0:
                error_participant_confidence.append(y)
                error_participant_feedback.append(X['feedback_t-1'])
            else:
                # Collect coefficients
                if cond == 'pos':
                    pos_coefficient.append(model.coef_)
                if cond == 'neg':
                    neg_coefficient.append(model.coef_)
        if len(pos_coefficient) > 0 and len(neg_coefficient) > 0:
            optimism = pos_coefficient[0][0] /  (pos_coefficient[0][0]
                                                 + neg_coefficient[0][0])
            optimisms.append(optimism)
            bdi = participant_data.bdi.unique()
            BDIs.append(bdi)
            PIDs.append(pid)
            BDI_level.append(participant_data.bdi_level.unique())
        else:
            pass

df_plot = pd.DataFrame({'optimism':optimisms,
                        'bdi': BDIs,
                        'pid': PIDs,
                        'bdi_level': BDI_level})

fig = plt.figure(figsize=(5,4))
ax = plt.subplot(1,1,1)
#plt.scatter(x=df_plot.optimism, y=df_plot.bdi)

plt.hist(df_plot[df_plot.bdi_level=='low'].optimism, alpha=0.5, bins=10)
plt.axvline(df_plot[df_plot.bdi_level=='low'].optimism.mean(), ls='--', c='C0',
            label=f"mean={round(df_plot[df_plot.bdi_level=='low'].optimism.mean(),2)}")
plt.hist(df_plot[df_plot.bdi_level=='high'].optimism, alpha=0.5, bins=60)
plt.axvline(df_plot[df_plot.bdi_level=='high'].optimism.mean(), ls='--',
            c='C1',
            label=f"mean={round(df_plot[df_plot.bdi_level=='high'].optimism.mean(),2)}")
plt.xlabel('Coefficient Asymmetry')
plt.ylabel('Count')
plt.legend(fontsize=(10))
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%%  Compute prediction error instead of feedback
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from matplotlib.ticker import FixedLocator
import statsmodels.api as sm
import matplotlib

# Set up DataFrame
df = expanded_df.copy()



# Compute the Prediction Error
#df['prediction_error'] = df['feedback'] - df['confidence_task']
# Group by 'pid' and 'condition' and then apply the rolling mean


# =============================================================================
# # Unexpected prediction error
# df['prediction_error'] = abs(df['feedback'] - df['confidence_task'])
#
# df['expected_PE'] = df.groupby(['pid', 'condition'])['prediction_error'].transform(lambda x: x.ewm(span=5).mean())
#
# df['unexpected_PE'] = df['prediction_error'] - df['expected_PE']
#
# df['pos_prediction_error'] = df['unexpected_PE'].apply(lambda x:
#                                                           x if x >= 0 else 0)
# df['neg_prediction_error'] = df['unexpected_PE'].apply(lambda x:
#                                                           x if x < 0 else 0)
# =============================================================================

# Prediction error
df['prediction_error'] = df['feedback'] - df['confidence_task']

df['pos_prediction_error'] = df['prediction_error'].apply(lambda x:
                                                          x if x >= 0 else 0)
df['neg_prediction_error'] = df['prediction_error'].apply(lambda x:
                                                          x if x < 0 else 0)

df['update'] = df['confidence_task'] + df['prediction_error']
df['PE_pos_update'] = np.where(df['prediction_error'] >= 0, df['update'], 0)
df['PE_neg_update'] = np.where(df['prediction_error'] < 0, df['update'], 0)

# Creating lagged PE columns for 1 to n_lag trials back
n_lag = 1
n_lag = n_lag+1
for i in range(1, n_lag):
    # Grouping by pid ensures participants do not share history
    df[f'pos_PE_t-{i}'] = df.groupby(['pid', 'condition'])['PE_pos_update'].shift(i)
    df[f'neg_PE_t-{i}'] = df.groupby(['pid', 'condition'])['PE_neg_update'].shift(i)

# Remove rows with NaN values
df.dropna(inplace=True)

# Set up cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=1)

error_participant_confidence = []
error_participant_feedback = []

all_coeffs = []
all_BDIs = []

# Normalize 'bdi' values to [0, 1] for color mapping
norm = Normalize(vmin=df['bdi'].min(), vmax=df['bdi'].max())
# Create a colormap from blue to red
cmap = matplotlib.colormaps.get_cmap('Reds')  # 'coolwarm' is a good blue to red cmap

# Initiate figure
fig = plt.figure(figsize=(7,7))

for bdi_nr, (bdi_level, color) in enumerate(zip(df.bdi_level.unique(),
                                                ['green', 'red'])):
    print(bdi_level)
    df_bdi = df[df.bdi_level==bdi_level]

    for i, cond in enumerate(df_bdi.condition.unique()):

        # Set up condition dataframe
        df_cond = df_bdi[df_bdi.condition==cond]

        coefficients = []
        BDIs = []

        for pid_nr, pid in enumerate(df_cond['pid'].unique()):
            participant_data = df_cond[df_cond['pid'] == pid]

            # Check if there is both pos and neg PEs
            if (all(participant_data['pos_PE_t-1'] == 0) or
                all(participant_data['neg_PE_t-1'] == 0)):
                break

            # Check if variance in confidence
            if (all(participant_data['confidence_task']==
                    participant_data['confidence_task'].values[0])):
                break

            X = participant_data[['pos_PE_t-1', 'neg_PE_t-1']]
            y = participant_data['confidence_task']

            # Add a constant to the model (statsmodels does not add it by default)
            X = sm.add_constant(X)

            # Initialize and fit the OLS model
            model = sm.OLS(y, X).fit()

            # Check if any coefficient is significant
            if any(model.pvalues[1:] < 0.05):  # We skip the constant's p-value
                #print(f'sig {model.params[1:].values}')
                coefficients.append(model.params[1:].values)  # Collect coefficients, excluding constant
                BDIs.append(participant_data['bdi'].iloc[0])  # Assuming 'bdi' is not unique per pid


# =============================================================================
#         # Prepare to collect coefficients
#         coefficients = []
#         BDIs = []
#         # Analyze each participant separately
#         for pid in df_cond['pid'].unique():
#             participant_data = df_cond[df_cond['pid'] == pid]
#             X = participant_data[['pos_PE_t-1',
#                                   'neg_PE_t-1']]
#             y = participant_data['confidence_task']
#
#             # Initialize and fit the model
#             model = LinearRegression()
#             model.fit(X, y)
#
#             if max(abs(model.coef_)) > 10:
#                 continue
#             else:
#                 # Collect coefficients
#                 coefficients.append(model.coef_)
#                 BDIs.append(participant_data.bdi.unique())
# =============================================================================

        # Convert to numpy array for analysis
        if len(coefficients) == 0:
            break
        else:
            coefficients = np.array(coefficients)

# =============================================================================
#         # Exclude outliers
#         # Calculate Q1 and Q3
#         data = coefficients
#         Q1 = np.percentile(data, 25, axis=0)
#         Q3 = np.percentile(data, 75, axis=0)
#
#         # Calculate the IQR
#         IQR = Q3 - Q1
#
#         # Determine outliers and clean data
#         lower_fence = Q1 - 1.5 * IQR
#         upper_fence = Q3 + 1.5 * IQR
#         outliers = (data < lower_fence) | (data > upper_fence)
#
#         # Create a mask for non-outliers
#         non_outliers = ~outliers.any(axis=1)
#
#         # Filter data to keep only non-outliers
#         coefficients = data[non_outliers]
# =============================================================================

        # Asymmetry
        e = 0.0000000001
        asymmetry = coefficients[:,0] / ((coefficients[:,0] +
                                          coefficients[:,1]) + e)

# =============================================================================
#         # Exclude outliers
#         # Calculate Q1 and Q3
#         data = asymmetry
#         Q1 = np.percentile(data, 25)
#         Q3 = np.percentile(data, 75)
#
#         # Calculate the IQR
#         IQR = Q3 - Q1
#
#         # Determine outliers and clean data
#         lower_fence = Q1 - 1.5 * IQR
#         upper_fence = Q3 + 1.5 * IQR
#         outliers = (data < lower_fence) | (data > upper_fence)
#
#         # Create a mask for non-outliers
#         non_outliers = ~outliers
#
#         # Filter data to keep only non-outliers
#         asymmetry = data[non_outliers]
#         coefficients = coefficients[non_outliers]
#
#         # Remove participants with negative Coefficients
#         neg_PE_coeffs = (coefficients[:,0] < 0) | (coefficients[:,1] < 0)
#         coefficients = coefficients[~neg_PE_coeffs]
#         asymmetry = asymmetry[~neg_PE_coeffs]
#
# =============================================================================
        # Calculate mean and sem
        asymmetry_mean = np.mean(asymmetry)
        asymmetry_SEM = (np.std(asymmetry) /
                         np.sqrt(len(asymmetry)))


        # Calculate mean and SEM of coefficients
        coeff_mean = np.median(coefficients, axis=0)
        coeff_sem = np.std(coefficients, axis=0) / np.sqrt(len(coefficients))

        # Specify full condition name
        if cond == 'neut' and bdi_level=='low':
            condition = 'Neutral'
            color = 'darkgrey'
            plot = 1
        if cond == 'neut' and bdi_level=='high':
            condition = 'Neutral'
            color ='black'
            plot = 2

        if cond == 'neg' and bdi_level=='low':
            condition = 'Negative'
            color = 'red'
            plot = 1
        if cond == 'neg' and bdi_level=='high':
            condition = 'Negative'
            color = 'darkred'
            plot = 2

        if cond == 'pos' and bdi_level=='low':
            condition = 'Positive'
            color = 'lightgreen'
            plot = 1
        if cond == 'pos' and bdi_level=='high':
            condition = 'Positive'
            color = 'darkgreen'
            plot = 2

        # Plotting the mean and SEM of the coefficients
        ax = plt.subplot(3,2,plot)
        ax.plot(np.arange(2), coeff_mean, color=color,
                label=f'{condition}, {bdi_level} BDI')

        ax.errorbar(x=np.arange(2), y=coeff_mean, yerr=coeff_sem,
                    fmt='o', capsize=5, color=color)

        plt.fill_between(np.arange(2),
                         coeff_mean-coeff_sem,
                         coeff_mean+coeff_sem,
                         color=color,
                         alpha=0.5)

        plt.legend(fontsize=(10), loc='lower right')
        plt.xlabel('Predictors')
        plt.ylabel('Coefficients')
        plt.xticks(rotation=45)
        plt.xticks(np.arange(2), ['PE+', 'PE-'])
        ax.spines[['top', 'right']].set_visible(False)
        plt.ylim(-0.5, 0.6)

        ax2 = plt.subplot(3,2,plot+2)

        for coeff, bdi in zip(coefficients, BDIs):

            # Get the color from the colormap
            normalized_bdi_value = norm(bdi)
            color_bdi = cmap(normalized_bdi_value)
            ax2.plot(np.arange(2), coeff, color=color_bdi, lw=0.6)

        #plt.title(f'{condition}')
        plt.xlabel('Predictors')
        plt.ylabel('Coefficients')
        plt.xticks(rotation=45)
        plt.xticks(np.arange(2), ['PE+', 'PE-'])
        ax2.spines[['top', 'right']].set_visible(False)
        plt.ylim(-0.1, 1.1)
        # Conditional colorbar addition
        if plot == 2:
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2, orientation='vertical')
            cbar.set_label('BDI Level')

        ax3 = plt.subplot(3,2,plot+4)

        ax3.errorbar(x=i, y=asymmetry_mean, yerr=asymmetry_SEM,
                    fmt='o', capsize=5, color=color,
                    label=f'{condition}, {bdi_level} BDI')
        ax3.scatter(x=np.linspace(i-.25, i+.25, len(asymmetry)), y=asymmetry,
                    c=color, s=2, alpha=1)
        #plt.legend(fontsize=(10), loc='lower right')
        plt.xlabel('Conditions')
        plt.ylabel('Asymmetry')
        plt.xticks(rotation=45,)
        #ax3.set_xticklabels([f'{cond}'])
        #ax3.xaxis.set_major_locator(FixedLocator(ticks))  # Applying FixedLocator with defined tick positions
        #x3.set_xticklabels(conditions)  # Setting tick labels
        ax3.spines[['top', 'right']].set_visible(False)
        #plt.ylim(-3, 2.5)

        all_coeffs.append(coefficients)
        all_BDIs.append(BDIs)

plt.tight_layout()
plt.show()


#%%


