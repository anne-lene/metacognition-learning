# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:50:13 2024

@author: carll
"""

# Regression models on EXP 2.

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
os.environ['R_HOME'] = r'C:\Users\carll\anaconda3\envs\metacognition_and_mood\lib\R'
from src.utility_functions import add_session_column
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.formula.api as smf
import statsmodels.api as sm
#from pymer4.models import Lmer
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

#  Calculate mean absolute error per trial
df['abs_error'] = abs(df['estimate'] - df['correct'])
mean_error_df = df.groupby(['pid', 'trial'])['abs_error'].mean().reset_index()
mean_error_df.rename(columns={'abs_error': 'mean_abs_error'}, inplace=True)

#  Merge the mean absolute error back to the main DataFrame
df = df.merge(mean_error_df, on=['pid', 'trial'], how='left')

# Remove subtrials except last subtrial
df = df[df['subtrial'] == 3]
df['abs_error_last_subtrial'] = abs(df['estimate'] - df['correct'])

# Shift feedback to get previous trial feedback per participant
df['prev_feedback'] = df.groupby('pid')['feedback'].shift(1)
df['prev_feedback_over_50'] = df['prev_feedback'].apply(lambda x: x if x > 50 else 0)
df['prev_feedback_below_50'] = df['prev_feedback'].apply(lambda x: x if x <= 50 else 0)

# Shift previous estimate by participant
df['prev_confidence'] = df.groupby('pid')['confidence'].shift(1)

# Calculate prediction error as the difference between previous feedback and previous estimate
df['prediction_error'] = df['prev_feedback'] - df['prev_confidence']

# Separate positive and negative prediction errors
df['positive_prediction_error'] = df['prediction_error'].apply(lambda x: x if x > 0 else 0)
df['negative_prediction_error'] = df['prediction_error'].apply(lambda x: x if x < 0 else 0)

df['prev_condition'] = df.groupby('pid')['condition'].shift(1)
df['prev_condition_neut'] = (df['prev_condition'] == 'neut').astype(int)
df['prev_condition_pos'] = (df['prev_condition'] == 'pos').astype(int)
df['prev_condition_neg'] = (df['prev_condition'] == 'neg').astype(int)

# Drop any NaN values due to shifting or first trial
df = df.dropna(subset=['mean_abs_error', 'prev_feedback', 'confidence', 'prev_condition_neut', 'prev_condition_pos', 'prev_condition_neg'])

# Set forgetting factor lambda
forgetting_factor = 0.5  # Adjust based on desired decay rate

# Initialize choice influence
df['choice_influence'] = np.nan  # Placeholder for choice influence values

# Ensure choice_influence is initialized as a float column to avoid type conflicts
df['choice_influence'] = 0.0  # Explicitly set as float

# Compute choice influence with forgetting factor within each participant’s trials
for pid in df['pid'].unique():
    participant_df = df[df['pid'] == pid]  # Filter for each participant
    choice_influence = [0.0]  # Start with initial influence as float

    # Calculate choice influence based on the forgetting factor
    for t in range(1, len(participant_df)):
        influence = (forgetting_factor * participant_df['confidence'].iloc[t - 1] +
                     (1 - forgetting_factor) * choice_influence[-1])
        choice_influence.append(influence)

    # Assign calculated choice influence values to the main DataFrame for this participant
    df.loc[df['pid'] == pid, 'choice_influence'] = choice_influence

# List of participants to remove
participants_to_remove = df['pid'].unique()[[14, 26, 35, 41, 42]]
participants_with_100_confidence = df[df['confidence'] == 100]['pid'].unique()
# Filter out the specified participants
df = df[~df['pid'].isin(participants_with_100_confidence)]

# Fit the LMM
model = smf.mixedlm("confidence ~ mean_abs_error + prev_feedback + positive_prediction_error + negative_prediction_error + choice_influence + bdi_score",
                    df, groups=df["pid"],
                    re_formula="1",
                    )
result = model.fit()

# =============================================================================
# # Define the formula and fit the model with a binomial family (or others based on your data type)
# model = Lmer("confidence ~ mean_abs_error + prev_feedback + choice_influence + (1|pid)", family='binomial', data=df)
# result = model.fit()
# =============================================================================

# Get the residuals
df['residuals'] = result.resid
df['fitted_values'] = result.fittedvalues

# Output results
print(result.summary())

#%% CPD

# Fit the full LMM model
full_model_formula = "confidence ~ mean_abs_error + prev_feedback + positive_prediction_error"
full_model = smf.mixedlm(full_model_formula, df, groups=df["pid"], re_formula="1")
full_result = full_model.fit()

# Calculate Total Sum of Squares (TSS)
tss = np.sum((df['confidence'] - df['confidence'].mean()) ** 2)

# Calculate Sum of Squared Residuals (SSR) for full model using `resid`
ssr_full = np.sum(full_result.resid ** 2)

# Calculate total variance explained by the full model (ESS / TSS)
total_var_explained_full = 1 - (ssr_full / tss)

# Dictionary to store partial R^2 values
partial_r_squared = {}

# List of predictors
predictors = ['mean_abs_error', 'prev_feedback', 'positive_prediction_error']

# Compute partial R^2 for each predictor
for predictor in predictors:
    # Fit reduced model without the current predictor
    reduced_formula = "confidence ~ " + " + ".join([p for p in predictors if p != predictor])
    reduced_model = smf.mixedlm(reduced_formula, df, groups=df["pid"], re_formula="1")
    reduced_result = reduced_model.fit()

    # Calculate SSR for the reduced model using `resid`
    ssr_reduced = np.sum(reduced_result.resid ** 2)

    # Calculate variance explained by the reduced model
    total_var_explained_reduced = 1 - (ssr_reduced / tss)

    # Calculate partial R^2 for the current predictor
    partial_r_squared[predictor] = total_var_explained_full - total_var_explained_reduced

# Display partial R^2 results
for predictor, r2 in partial_r_squared.items():
    print(f"Partial R^2 for {predictor}: {r2:.4f}")

print(full_result.summary())

#%% Visualise fit
import matplotlib.pyplot as plt
import seaborn as sns

# Add the fitted values from the model to the DataFrame
df['fitted_confidence'] = result.fittedvalues

# Plot observed vs. fitted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x='confidence', y='fitted_confidence', data=df, alpha=0.6)
plt.plot([df['confidence'].min(), df['confidence'].max()],
         [df['confidence'].min(), df['confidence'].max()], color='gray', linestyle='--')
plt.xlabel('Observed Confidence')
plt.ylabel('Fitted Confidence')
plt.title('Observed vs. Fitted Confidence')
plt.show()
#%%  Visualise fit over time

# Select a subset of participants if there are many
sample_participants = df['pid'].unique()[:]  # Adjust the number as needed
# spikes: 14, 26, 35, 41, 42(opposite),
plt.figure(figsize=(14, 8))
for pid in sample_participants:
    participant_data = df[df['pid'] == pid]

    plt.plot(participant_data['trial'], participant_data['confidence'],
             label=f'Observed (PID {pid})', alpha=0.3)
    plt.plot(participant_data['trial'], participant_data['fitted_confidence'],
             linestyle='--', label=f'Fitted (PID {pid})')

plt.xlabel('Trial')
plt.ylabel('Confidence')
plt.title('Observed and Fitted Confidence Over Trials (Subset of Participants)')
plt.show()

#%% Residuals vs Fitted plot
from statsmodels.stats.diagnostic import het_breuschpagan

# Extract residuals and fitted values
residuals = result.resid
fitted_values = result.fittedvalues
plt.figure(figsize=(8, 6))
plt.scatter(fitted_values, residuals, alpha=0.5)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.show()

#%% Q-Q plot
import scipy.stats as stats
plt.figure(figsize=(8, 6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals')
plt.show()


#%% Test for heteroscedasticity
# Add a constant to the explanatory variables
explanatory_vars = sm.add_constant(df['fitted_values'])

# Perform Breusch-Pagan test
bp_test = het_breuschpagan(df['residuals'], explanatory_vars)
print(f"Breusch-Pagan test statistic: {bp_test[0]}, p-value: {round(bp_test[1],4)}")

#%% test for normality
from scipy.stats import shapiro

# Perform Shapiro-Wilk test
plt.hist(df['residuals'], bins=20)
plt.show()
shapiro_test = shapiro(df['residuals'])
print(f"Shapiro-Wilk test statistic: {shapiro_test.statistic}, p-value: {round(shapiro_test.pvalue, 4)}")



#%% color by participant
# Extract residuals and fitted values
residuals = result.resid
fitted_values = result.fittedvalues

# Assuming you have a DataFrame `df` with the fitted values, residuals, and participant IDs
df['fitted_values'] = fitted_values
df['residuals'] = residuals

# Plot with coloring by participant
plt.figure(figsize=(10, 6))
participants = df['pid'].unique()  # Get unique participant IDs
for participant in participants:
    participant_data = df[df['pid'] == participant]
    plt.scatter(participant_data['fitted_values'], participant_data['residuals'],
                alpha=0.5, label=f'Participant {participant}')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values Colored by Participant')
#plt.legend(title="Participant", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#%% Normal distribution of random effects?
import seaborn as sns
# Extract random effects
random_effects = result.random_effects
# Plot distribution of random effects
plt.figure(figsize=(8, 6))
sns.histplot([v[0] for v in random_effects.values()], kde=True)
plt.xlabel('Random Effect Values')
plt.title('Distribution of Random Effects')
plt.show()

#%% Autocorrelation in time?
from statsmodels.graphics.tsaplots import plot_acf
# Autocorrelation plot for residuals
plt.figure(figsize=(8, 6))
plot_acf(residuals, lags=14)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of Residuals')
plt.show()

#%% Try removing outliers

# Identify and remove outliers
# Calculate mean absolute residual for each participant
participant_residuals = df.groupby('pid')['residuals'].apply(lambda x: np.mean(np.abs(x))).reset_index()
participant_residuals.rename(columns={'residuals': 'mean_abs_residual'}, inplace=True)
# Calculate z-scores of the mean absolute residuals
participant_residuals['z_score'] = stats.zscore(participant_residuals['mean_abs_residual'])

# Define participants as outliers if their residual z-score exceeds a threshold, e.g., |z| > 2.5 or 3
outliers = participant_residuals[abs(participant_residuals['z_score']) > 3]['pid']
# Filter out outlier participants
df_filtered = df[~df['pid'].isin(outliers)]

# Fit the LMM
model = smf.mixedlm("confidence ~ mean_abs_error + prev_feedback + choice_influence",
                    df_filtered, groups=df_filtered["pid"],
                    re_formula="1")
result = model.fit()

# Output results
print(result.summary())

#%% Residuals vs Fitted plot
# Add the fitted values from the model to the DataFrame
df['fitted_confidence'] = result.fittedvalues

# Plot observed vs. fitted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x='confidence', y='fitted_confidence', data=df, alpha=0.6)
plt.plot([df['confidence'].min(), df['confidence'].max()],
         [df['confidence'].min(), df['confidence'].max()], color='gray', linestyle='--')
plt.xlabel('Observed Confidence')
plt.ylabel('Fitted Confidence')
plt.title('Observed vs. Fitted Confidence')
plt.show()