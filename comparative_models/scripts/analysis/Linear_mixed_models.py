# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:52:09 2024

@author: carll
"""

# Linear Mixed Models

import numpy as np
from src.utils import (add_session_column, load_df)
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
import scipy.stats as stats
import statsmodels.formula.api as smf



#%%
df_a = load_df(EXP=1)


#%% Add session column
df_a = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

#%% Restructure df

# Get bdi score for each participant
bdi_scores = df_a.groupby('pid')['bdi'].first().reset_index()


df = df_a.copy()


# Create trial data where only single numbers
df['trial'] = [np.array(range(20)) for i in range(len(df))]
df['session'] = [[i]*20 for i in df['session'].values]
df['pid'] = [[i]*20 for i in df['pid'].values]
df['condition'] = [[i]*20 for i in df['condition'].values]

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
expanded_df = pd.merge(expanded_df, bdi_scores, on='pid', how='left')


#%% LMM: Confidence ~ previous_feedback + error

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

df['previous_pavg_task'] = df.groupby(['pid',
                                       'session']
                                      )['p_avg_task'].shift(1)

# Create a mask that identifies the first row in each group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0

# Use the mask to filter out the first row of each group
df = df[~is_first_row].reset_index(drop=True)

# Unique conditions in the DataFrame
conditions = df['condition'].unique()

# Define the model formula
model_formula = 'confidence_task ~ previous_feedback + error'

# Fit the mixed model with 'pid' as a random effect
mixed_model = smf.mixedlm(model_formula, df,
                          #re_formula="1 + error",
                          groups=df['pid']).fit()

print(mixed_model.summary())

# Get the residuals
residuals = mixed_model.resid
df['residuals'] = residuals
grouped_residuals = df.groupby('pid')['residuals'].sum()

metacogbias = []
participants = []
for participant, effects in mixed_model.random_effects.items():
    intercept = effects['Group']
   # slope = effects['error']  # Accessing the varying slope for 'performance'
    metacogbias.append(intercept)
    participants.append(participant)

# Group by participant and calculate the sum of residuals for each participant
residuals_sum = df.groupby('pid')['residuals'].sum().reset_index()
residuals_sum.rename(columns={'residuals': 'sum_residuals'}, inplace=True)
df = pd.merge(df, residuals_sum, on='pid', how='left')

# Add bdi score of each participant
df = pd.merge(df, pd.DataFrame({'pid':participants,
                                'metacogbias': metacogbias}),
              on='pid', how='left')


#%% Correlate metacognitive bias with BDI

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Assuming 'df' is your DataFrame and it contains 'bdi' and 'metacogbias' columns
# df = pd.DataFrame({'bdi': [...], 'metacogbias': [...]})

# Calculate Pearson correlation and p-value
pearson_r, p_value = stats.pearsonr(df['bdi'], df['metacogbias'])

fig, axs = plt.subplots(1,1, figsize=(5,5))
# Create a scatter plot
plt.scatter(df['bdi'], df['metacogbias'], color='blue', alpha=0.5,
            label='Data points',)

# Fit a linear regression trendline
slope, intercept, r_value, p_value, std_err = stats.linregress(df['bdi'],
                                                               df['metacogbias'])
line = slope * df['bdi'] + intercept
plt.plot(df['bdi'], line, color='red',
         label=f'Trendline (r={pearson_r:.2f}, p={p_value:.3f})')

# Annotate the plot with the Pearson correlation coefficient and p-value
#plt.annotate(f'r={pearson_r:.2f}, p={p_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
#             ha='left', va='top', fontsize=10, color='red')

# Add labels and legend
plt.xlabel('BDI')
plt.ylabel('Metacognitive Bias')
plt.legend(fontsize=10)
axs.spines[['top', 'right']].set_visible(False)

# Show the plot
plt.show()


#%% LMM: Confidence ~ error

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

df['previous_pavg_task'] = df.groupby(['pid',
                                       'session']
                                      )['p_avg_task'].shift(1)

# Create a mask that identifies the first row in each group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0

# Use the mask to filter out the first row of each group
df = df[~is_first_row].reset_index(drop=True)

# Unique conditions in the DataFrame
conditions = df['condition'].unique()

# Define the model formula
model_formula = 'confidence_task ~ error'

# Fit the mixed model with 'pid' as a random effect
mixed_model = smf.mixedlm(model_formula, df,
                          #re_formula="1 + error",
                          groups=df['pid']).fit()

print(mixed_model.summary())

# Get the residuals
residuals = mixed_model.resid
df['residuals'] = residuals
grouped_residuals = df.groupby('pid')['residuals'].sum()

metacogbias = []
participants = []
for participant, effects in mixed_model.random_effects.items():
    intercept = effects['Group']
   # slope = effects['error']  # Accessing the varying slope for 'performance'
    metacogbias.append(intercept)
    participants.append(participant)

# Group by participant and calculate the sum of residuals for each participant
residuals_sum = df.groupby('pid')['residuals'].sum().reset_index()
residuals_sum.rename(columns={'residuals': 'sum_residuals'}, inplace=True)
df = pd.merge(df, residuals_sum, on='pid', how='left')

# Add bdi score of each participant
df = pd.merge(df, pd.DataFrame({'pid':participants,
                                'metacogbias': metacogbias}),
              on='pid', how='left')


#%% Calculate mean confidence_task - previous feedback

df['mean_confidence'] = df.groupby(['pid', 'session'])['confidence_task'].transform('mean')
df['feedback_aligment'] = df['mean_confidence'] - df['previous_feedback']
df['avg_feedback_aligment']  = df.groupby(['pid'])['feedback_aligment'].transform('mean')

#%% Correlate avg feedback aligment with BDI

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


fig, axs = plt.subplots(1,1, figsize=(5,5))
# Create a scatter plot
plt.scatter(df['bdi'], df['avg_feedback_aligment'], color='blue', alpha=0.5,
            label='Data points',)

# Fit a linear regression trendline
slope, intercept, r_value, p_value, std_err = stats.linregress(df['bdi'],
                                                 df['avg_feedback_aligment'])
line = slope * df['bdi'] + intercept
plt.plot(df['bdi'], line, color='red',
         label=f'Trendline (r={pearson_r:.2f}, p={p_value:.3f})')

# Add labels and legend
plt.xlabel('BDI')
plt.ylabel('Avg. feedback aligment')
plt.legend(fontsize=10)
axs.spines[['top', 'right']].set_visible(False)

# Show the plot
plt.show()


#%% Correlate avg feedback aligment with metacognitive bias

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Assuming 'df' is your DataFrame and it contains 'bdi' and 'metacogbias' columns
# df = pd.DataFrame({'bdi': [...], 'metacogbias': [...]})

# Calculate Pearson correlation and p-value
pearson_r, p_value = stats.pearsonr(df['metacogbias'], df['avg_feedback_aligment'])

fig, axs = plt.subplots(1,1, figsize=(5,5))
# Create a scatter plot
plt.scatter(df['metacogbias'], df['avg_feedback_aligment'], color='blue', alpha=0.5,
            label='Data points',)

# Fit a linear regression trendline
slope, intercept, r_value, p_value, std_err = stats.linregress(df['metacogbias'],
                                                 df['avg_feedback_aligment'])
line = slope * df['metacogbias'] + intercept
plt.plot(df['metacogbias'], line, color='red',
         label=f'Trendline (r={pearson_r:.2f}, p={p_value:.3f})')

# Add labels and legend
plt.xlabel('Metacognitive bias')
plt.ylabel('Avg. feedback aligment')
plt.legend(fontsize=10)
axs.spines[['top', 'right']].set_visible(False)

# Show the plot
plt.show()

#%% regression

fig, axs = plt.subplots(1,1, figsize=(5,5))

# Create a scatter plot
plt.scatter(df['bdi'], df['avg_feedback_aligment'], color='blue', alpha=0.5,
            label='Data points')

# Fit a linear regression trendline
slope, intercept, r_value, p_value, std_err = stats.linregress(df['bdi'], df['avg_feedback_aligment'])
line = slope * df['bdi'] + intercept

# Plot the regression line
plt.plot(df['bdi'], line, color='red', label=f'Trendline: y={slope:.2f}x+{intercept:.2f}\n(r={r_value:.2f}, p={p_value:.3f})')

# Add labels and legend
plt.xlabel('BDI')
plt.ylabel('Mean confidence - feedback')
plt.legend(fontsize=10)
axs.spines[['top', 'right']].set_visible(False)

# Show the plot
plt.show()


#%%

from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from scipy.stats import pearsonr

plt.figure(figsize=(7, 6))

# Assuming 'df' is your DataFrame and includes the predictors
predictors_df = df[['metacogbias', 'avg_feedback_aligment']]

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

plt.show()

#%%
import seaborn as sns

lmm_formula = "bdi ~ avg_feedback_aligment + metacogbias"

lmm_model = smf.mixedlm(lmm_formula, df, groups=df["pid"],
                        re_formula="1 + avg_feedback_aligment + metacogbias"
                        )
lmm_result = lmm_model.fit()

print(lmm_result.summary())



#%% condition wise: Calculate mean confidence_task - previous feedback

df['mean_confidence'] = df.groupby(['pid', 'session'])['confidence_task'].transform('mean')
df['feedback_aligment'] = df['mean_confidence'] - df['previous_feedback']
df['avg_feedback_aligment']  = df.groupby(['pid', 'session'])['feedback_aligment'].transform('mean')

#%% Correlate avg feedback aligment with BDI

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats


fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(12,4))

for ax, cond, color in zip([ax1, ax2, ax3], ['neut', 'pos', 'neg'], ['grey', 'green', 'red']):
    df_cond = df[df.condition==cond]
    # Create a scatter plot
    ax.scatter(df_cond['bdi'], df_cond['avg_feedback_aligment'], alpha=0.5,
                label='Data points', color=color)

    # Fit a linear regression trendline
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_cond['bdi'],
                                                     df_cond['avg_feedback_aligment'])
    line = slope * df_cond['bdi'] + intercept
    ax.plot(df_cond['bdi'], line, color='k',
             label=f'Trendline (r={pearson_r:.2f}, p={p_value:.3f})',
             lw=3
             )

    # Add labels and legend
    ax.set_xlabel('BDI')
    ax.set_ylabel('Avg. feedback aligment')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(-61, 71)

# Show the plot
plt.tight_layout()
plt.show()

