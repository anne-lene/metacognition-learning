# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 00:03:16 2024

@author: carll
"""

# Clustering as a filter for similar update rules


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
import umap


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

#%% Cluser z-scored data

from sklearn.cluster import SpectralClustering
import seaborn as sns
import os
import hdbscan
from scipy.signal import savgol_filter
os.environ['OMP_NUM_THREADS'] = '2'  # Set to 2 or however many threads you want to allow

# Now import KMeans
from sklearn.cluster import KMeans

# Set up DataFrame
df = expanded_df.copy()

# Group by participant and calculate mean and standard deviation for 'confidence_task'
confidence_stats = df.groupby(['pid', 'session'])['confidence_task'].agg(['mean', 'std'])

# Join these statistics back to the original DataFrame
df = df.join(confidence_stats, on=['pid', 'session'], rsuffix='_conf_stats')

# Calculate Z-scores for 'confidence_task'
df['confidence_task_z'] = (df['confidence_task'] - df['mean']) / df['std']

# Apply the same Z-score transformation to 'feedback'
df['feedback_z'] = (df['feedback'] - df['mean']) / df['std']

# Adjust feedback to the transformed scale using mean and standard deviation of confidence_task
df['feedback_transformed'] = df['confidence_task_z'] * df['std'] + df['mean']

# Calculate the mean of feedback_z for each pid and condition combination
df['mean_feedback'] = df.groupby(['pid', 'condition'])['feedback'].transform('mean')

# Subtract confidence_task_z from this mean
df['confidence_task_z'] = df['mean_feedback'] - df['confidence_task']




# Drop the extra columns if they are no longer needed
df.drop(['mean', 'std'], axis=1, inplace=True)


# Apply smoothing
window_length = 5  # Needs to be an odd number, depends on your data's sampling rate
polyorder = 2  # Polynomial order: 2 or 3 usually works well

# Apply Savitzky-Golay filter
df['confidence_task_z'] = df.groupby(['pid',
                                               'condition'])['confidence_task_z'].transform(
    lambda x: savgol_filter(x, window_length=window_length,
                            polyorder=polyorder, mode='nearest'))

# Reshape data

# Pivot the DataFrame so each participant's trials are in one row
df_pivot = df.pivot_table(index=['pid', 'condition'], columns='trial',
                          values='confidence_task_z')

# Example using Spectral Clustering
n_clusters = 3  # You may need to determine the best number of clusters through experimentation
sc = SpectralClustering(n_clusters=n_clusters,
                        affinity='nearest_neighbors')

# =============================================================================
# sc = hdbscan.HDBSCAN(min_cluster_size=3,
#                      min_samples=1,
#                      gen_min_span_tree=True)
# =============================================================================

# Fit the model on the pivoted data
clusters = sc.fit_predict(df_pivot)

# Add cluster assignments back to the original DataFrame
#df['cluster'] = df['pid'].map(dict(zip(df_pivot.index, clusters)))

# Assign clusters back to the original DataFrame (if needed for further analysis)
df['cluster'] = list(np.repeat(clusters, df.groupby(['pid',
                                                     'condition']).size()))

# Unique colors for clusters
colors = sns.color_palette("hsv", len(df['cluster'].unique()))

#%
# Plot each participant's session
fig, [ax1, ax2, ax3] = plt.subplots(1,3, figsize=(12, 3))
for (pid, condition), cluster, in (zip(df_pivot.index, clusters,
                                       )):

    if condition == 'neut':
        ax = ax1
    if condition == 'pos':
        ax = ax2
    if condition == 'neg':
        ax = ax3

    data = df_pivot.loc[(pid, condition)].values
    ax.plot(data, marker='o', linestyle='-', color=colors[cluster],
            label=f'PID {pid} Condition {condition} Cluster {cluster}',
            lw=1,
            markersize=1)

    # Simplify the legend (optional)
    #handles, labels = plt.gca().get_legend_handles_labels()
    #by_label = dict(zip(labels, handles))  # Unique labels
    #plt.legend(by_label.values(), by_label.keys())

    ax.set_title(f'{condition}')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Z-scored Confidence Task')
    ax.spines[['top', 'right']].set_visible(False)

plt.show()


#%% First Umap

from sklearn.cluster import SpectralClustering
import seaborn as sns
import os
import hdbscan
import umap
from sklearn.preprocessing import StandardScaler

os.environ['OMP_NUM_THREADS'] = '2'  # Set to 2 or however many threads you want to allow


# DO Umap

# Reshape data
df_short = df[['pid', 'condition', 'confidence_task_z', 'trial']]
# Pivot the DataFrame so each participant's trials are in one row
df_pivot = df_short.pivot_table(index=['pid', 'condition'], columns='trial',
                          values='confidence_task_z')

# Standardize the data if needed
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df_pivot)  # Assuming first two columns are 'pid' and 'session'

# Initialize UMAP
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

# Fit and transform the data
embedding = reducer.fit_transform(data_scaled)

# Add UMAP dimensions back to the original DataFrame (or a summary DataFrame)
df_pivot['UMAP1'] = embedding[:, 0]
df_pivot['UMAP2'] = embedding[:, 1]

#%%

# Ensure that the index is reset if you want to use columns in seaborn plots directly
df_plot = df_pivot.reset_index()

colors = sns.color_palette(['grey', 'red', 'green'],
                           len(df['cluster'].unique()))


# Plotting using seaborn
plt.figure(figsize=(10, 8))
ax = plt.subplot(1,1,1)
sns.scatterplot(x='UMAP1', y='UMAP2', hue='condition',
                #style='pid',
                palette=colors,#'viridis',
                data=df_plot, s=100, alpha=0.6,
                legend=False)
plt.title('UMAP Projection of Confidence Task Z-Scores by Session')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
ax.spines[['top', 'right']].set_visible(False)
#plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.grid(True)
plt.show()

#%% One plot per condition

# Get unique conditions
conditions = df_plot['condition'].unique()

# Setting up the plotting environment
fig, axes = plt.subplots(nrows=len(conditions), figsize=(7, 5 * len(conditions)))  # Adjust the size as necessary

for i, cond in enumerate(conditions):
    # Filter data for the current condition
    cond_data = df_plot[df_plot['condition'] == cond]

    if cond=='neut':
        color=colors[0]
    if cond=='neg':
        color=colors[1]
    if cond=='pos':
        color=colors[2]

    # Create a scatter plot for each condition
    ax = axes[i] if len(conditions) > 1 else axes
    sns.scatterplot(x='UMAP1', y='UMAP2',
                    hue='condition',
                    palette='viridis',
                    data=cond_data,
                    ax=ax, s=100,
                    alpha=0.6, legend=False)

    # Set plot titles and labels
    ax.set_title(f'UMAP Projection for Condition: {cond}')
    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')
    ax.spines[['top', 'right']].set_visible(False)
    #ax.legend(title='PID', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout
plt.tight_layout()
plt.show()


#%% HDBSCAN on UMAP

# Prepare a DataFrame to store cluster labels
df_plot['cluster'] = -1  # Default label for noise

# Cluster data within each condition
for condition in df_plot['condition'].unique():
    # Filter the DataFrame for the current condition
    data_condition = df_plot[df_plot['condition'] == condition]

    # Apply HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1)
    cluster_labels = clusterer.fit_predict(data_condition[['UMAP1', 'UMAP2']])

    # Store the cluster labels in the main DataFrame
    df_plot.loc[df_plot['condition'] == condition, 'cluster'] = cluster_labels

#%%

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='white')

# Define the conditions explicitly if they are known
conditions = ['neut', 'neg', 'pos']  # Replace 'cond1', 'cond2', 'cond3' with your actual conditions

# Create a large figure to accommodate the subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows and 3 columns

for i, condition in enumerate(conditions):
    # Index for the top row (Trajectories)
    ax_traj = axes[0, i]
    # Index for the bottom row (UMAP)
    ax_umap = axes[1, i]

    # Filter data for the current condition for trajectory plot
    data_to_plot = df[(df['condition'] == condition)]
    clusters = df_plot[df_plot['condition'] == condition]['cluster']

    # Mapping clusters to colors for consistency in both plots
    unique_clusters = np.unique(clusters)
    palette = sns.color_palette('viridis', len(unique_clusters))
    cluster_color_map = {cluster: palette[j]
                         for j, cluster
                         in enumerate(unique_clusters)}

    # Trajectory plot
    pids = data_to_plot['pid'].unique()
    for pid in pids:
        participant_data = data_to_plot[data_to_plot['pid'] == pid]
        cluster = df_plot[(df_plot['pid'] == pid) &
                          (df_plot['condition'] ==
                           condition)]['cluster'].iloc[0]
        sns.lineplot(x='trial', y='confidence_task_z', data=participant_data,
                     ax=ax_traj, color=cluster_color_map[cluster],
                     legend=False)

    ax_traj.set_title(f'Trajectories in {condition}')
    ax_traj.set_xlabel('Trial Number')
    ax_traj.set_ylabel('Z-scored Confidence Task')
    ax_traj.spines[['top', 'right']].set_visible(False)

    # UMAP plot
    df_condition = df_plot[df_plot['condition'] == condition]
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster',
                    palette=palette,
                    data=df_condition, ax=ax_umap,
                    legend=False, s=100, alpha=0.6)

    ax_umap.set_title(f'UMAP Projection for {condition}')
    ax_umap.set_xlabel('UMAP Dimension 1')
    ax_umap.set_ylabel('UMAP Dimension 2')
    ax_umap.spines[['top', 'right']].set_visible(False)
   # ax_umap.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust the layout to prevent overlap
plt.grid(False)
plt.tight_layout()
plt.show()


#%% What Factors Predict Confidence? Linear regression

import pandas as pd
import statsmodels.api as sm

# Set up DataFrame
df = expanded_df.copy()

# shift down by one to get 'previous_x'
df['previous_feedback'] = df.groupby(['pid',
                                      'session']
                                     )['feedback'].shift(1)

df['previous_error'] = df.groupby(['pid',
                                   'session']
                                  )['error'].shift(1)

df['previous_confidence'] = df.groupby(['pid',
                                   'session']
                                  )['confidence_task'].shift(1)

df['condition_nr'] = df['condition'].replace('neut', 0).replace('neg', -1).replace('pos', 1)


# Create a mask that identifies the first row in each group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0

# Use the mask to filter out the first row of each group
df = df[~is_first_row].reset_index(drop=True)

# Initialize a dictionary to store results
results_dict = {'pid': [], 'Variable': [], 'Coefficients': [], 'P-values': [],
                'bdi': []}

for pid in df['pid'].unique():
    df_pid = df[df['pid'] == pid]

    # Defining the predictor variables and the response variable
    X = df_pid[['error', 'trial', 'previous_feedback', 'previous_confidence', 'session', 'condition_nr']]
    X = sm.add_constant(X)  # Adding a constant to the model (intercept)
    y = df_pid['confidence_task']

    # Creating the linear regression model
    model = sm.OLS(y, X).fit()

    # Getting the summary of the model
    #print(model.summary())

    # Extracting coefficients and p-values
    variables = model.params.index.tolist()
    coefficients = model.params.values
    p_values = model.pvalues.values
    pids = [pid] * len(variables)
    bdi = [df_pid.bdi.unique()[0]] * len(variables)

    # Store results in the dictionary
    results_dict['pid'].extend(pids)
    results_dict['Variable'].extend(variables)
    results_dict['Coefficients'].extend(coefficients)
    results_dict['P-values'].extend(p_values)
    results_dict['bdi'].extend(bdi)


# Convert the dictionary to DataFrame
df_results = pd.DataFrame(results_dict)

# Save the combined results to a CSV file
df_results.to_csv('regression_results.csv', index=False)
print("All participants' coefficients and p-values have been saved to regression_results.csv.")

#%%

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Set up DataFrame
df = expanded_df.copy()

# shift down by one to get 'previous_x'
df['previous_feedback'] = df.groupby(['pid', 'session'])['feedback'].shift(1)
df['previous_error'] = df.groupby(['pid', 'session'])['error'].shift(1)
df['previous_confidence'] = df.groupby(['pid', 'session'])['confidence_task'].shift(1)
df['condition_nr'] = df['condition'].replace({'neut': 0, 'neg': -1, 'pos': 1})

# Create a mask that identifies the first row in each group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0

# Use the mask to filter out the first row of each group
df = df[~is_first_row].reset_index(drop=True)

# Initialize a dictionary to store results
results_dict = {'pid': [], 'Variable': [],
                'Coefficients':[], #'Partial R^2': [],
                'P-values': [], 'bdi': []}

for pid in df['pid'].unique():
    df_pid = df[df['pid'] == pid]

    # Defining the predictor variables and the response variable
    X = df_pid[['error', 'trial', 'previous_feedback', 'previous_confidence', 'session', 'condition_nr']]
    X = sm.add_constant(X)  # Adding a constant to the model (intercept)
    y = df_pid['confidence_task']

    # Creating the full model
    full_model = sm.OLS(y, X).fit()

    # For each predictor, calculate the partial R^2
    for var in X.columns:
        if var != 'const':  # Skip the intercept
            reduced_model = sm.OLS(y, X.drop(columns=[var])).fit()
            ssr_full = full_model.ssr
            ssr_reduced = reduced_model.ssr
            partial_r_squared = 1 - (ssr_full / ssr_reduced)
            results_dict['pid'].append(pid)
            results_dict['Variable'].append(var)
            #results_dict['Partial R^2'].append(partial_r_squared)
            results_dict['Coefficients'].append(partial_r_squared)
            results_dict['P-values'].append(full_model.pvalues[var])
            results_dict['bdi'].append(df_pid.bdi.unique()[0])

# Convert the dictionary to DataFrame
df_results = pd.DataFrame(results_dict)

# Save the combined results to a CSV file
df_results.to_csv('regression_results.csv', index=False)
print("All participants' partial R^2 and p-values have been saved to regression_results.csv.")


#%% Get P-value of partial r2 from permutation test
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(X):
    VIF = pd.DataFrame()
    if 'const' in X.columns:
        X = X.drop(['const'], axis=1)
    VIF["variable"] = X.columns
    VIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return VIF

# Set up DataFrame
df = expanded_df.copy()

# shift down by one to get 'previous_x'
df['previous_feedback'] = df.groupby(['pid', 'session'])['feedback'].shift(1)
df['previous_error'] = df.groupby(['pid', 'session'])['error'].shift(1)
df['previous_confidence'] = df.groupby(['pid', 'session'])['confidence_task'].shift(1)
df['condition_nr'] = df['condition'].replace({'neut': 0, 'neg': -1, 'pos': 1})

# Filter the DataFrame to avoid the first trial in each session/group
is_first_row = df.groupby(['pid', 'session']).cumcount() == 0
df = df[~is_first_row].reset_index(drop=True)

# Define the number of permutations
n_permutations = 1000
results = []

# Iterate over each participant
for pid in tqdm(df['pid'].unique(), total=len(df['pid'].unique())):
    df_pid = df[df['pid'] == pid]
    y = df_pid['confidence_task']
    X = df_pid[['error',
                'trial',
                'previous_feedback',
                #'previous_confidence',
                #'session',
                #'condition_nr',
                ]]
    X = sm.add_constant(X)

    # Check VIF before fitting the model
    vif_data = calculate_vif(X)
    if vif_data['VIF'].max() > 5:  # Choose a threshold suitable for your analysis
        print(f"High multicollinearity detected for PID {pid} with VIF: \n{vif_data}")
        continue  # Optionally skip fitting this model, or handle it differently

    # Fit the full model with the actual data
    full_model = sm.OLS(y, X).fit()
    original_ssr = full_model.ssr

    # Compute the original partial R^2
    original_partial_r2 = {}
    for var in X.columns:
        if var != 'const':  # Skip the intercept
            reduced_model = sm.OLS(y, X.drop(columns=[var])).fit()
            ssr_reduced = reduced_model.ssr
            partial_r2 = 1 - (original_ssr / ssr_reduced)
            original_partial_r2[var] = partial_r2

    # Permutation test
    permuted_p_values = {var: 0 for var in X.columns if var != 'const'}
    for _ in range(n_permutations):
        y_permuted = np.random.permutation(y)
        full_model_perm = sm.OLS(y_permuted, X).fit()
        ssr_perm = full_model_perm.ssr

        for var in X.columns:
            if var != 'const':
                reduced_model_perm = sm.OLS(y_permuted, X.drop(columns=[var])).fit()
                ssr_reduced_perm = reduced_model_perm.ssr
                partial_r2_perm = 1 - (ssr_perm / ssr_reduced_perm)
                if partial_r2_perm >= original_partial_r2[var]:
                    permuted_p_values[var] += 1

    # Calculate p-values
    for var in permuted_p_values:
        permuted_p_values[var] /= n_permutations
        results.append({'pid': pid,
                        'variable': var,
                        'partial_r2': original_partial_r2[var],
                        'permuted_p_value': permuted_p_values[var],
                        'bdi': df_pid.bdi.unique()[0]})


#%%
# Convert results to DataFrame and save or print
df_results = pd.DataFrame(results)

# Save the combined results to a CSV file
df_results.to_csv('regression_results_trials.csv', index=False)
print("All participants' partial R^2 and p-values have been saved to regression_results.csv.")

#%% Plot results
import matplotlib.pyplot as plt

# Filter out the intercept and only include p-values < 0.05
alpha = 0.05
#df_significant_pvalues = df_results[(df_results['P-values'] < alpha) & (df_results['Variable'] != 'const')]
df_significant_pvalues = df_results[(df_results['permuted_p_value'] < alpha) & (df_results['variable'] != 'const')]

# Count the number of occurrences for each predictor
frequency = df_significant_pvalues['variable'].value_counts().sort_index()

# Plotting the bar plot
plt.figure(figsize=(7, 5))
ax = plt.subplot(1,1,1)
frequency.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title(f'Frequency of P-values < {alpha} for Each Predictor')
#plt.xlabel('Predictor')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()  # Adjusts plot to ensure everything fits without overlap
ax.spines[['top', 'right']].set_visible(False)
plt.show()


#%%
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt


# Filter out the intercept and only include p-values < 0.05
alpha = 0.05
df_results_filt = df_results.copy()
df_results_filt = df_results_filt[df_results_filt['variable'] != 'const']
df_results_filt = df_results_filt.groupby('pid').filter(lambda x: (x['permuted_p_value'] < alpha).any())
df_results_filt.drop(columns=['bdi'], inplace=True)


# Pivoting the DataFrame to get one row per participant with each coefficient as a column
df_pivot = df_results_filt.pivot(index='pid', columns='variable', values='partial_r2')

# Fill any missing values if necessary (depends on your data)
df_pivot.fillna(0, inplace=True)

# Normalize each participant's coefficients vector between 0 and 1
#df_pivot = df_pivot.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# UMAP Dimensionality Reduction
reducer = umap.UMAP(n_neighbors=15, random_state=42)
umap_embedding = reducer.fit_transform(df_pivot)

# Prepare the plotting environment
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# UMAP Plot
ax[0].scatter(umap_embedding[:, 0], umap_embedding[:, 1], alpha=0.7)
ax[0].set_title('UMAP Projection of Participant Partial r2')
ax[0].set_xlabel('UMAP Dimension 1')
ax[0].set_ylabel('UMAP Dimension 2')
ax[0].spines[['top', 'right']].set_visible(False)

# Coefficients Plot for a specific participant (example: the first participant)
for i in range(len(df_pivot.index)):
    selected_pid = df_pivot.index[i]
    ax[1].plot(df_pivot.columns, df_pivot.loc[selected_pid])
ax[1].set_title('Partial r2 for Participant')
ax[1].set_xlabel('Variable')
ax[1].set_ylabel('Partial_r2 Value')
ax[1].tick_params(axis='x', rotation=45)
ax[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

#%% Cluster using HDBSCAN

import pandas as pd
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler


# Set up df
df_results_filt = df_results.copy()

include_bdi = False

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit and transform the data
df_results_filt['bdi'] = scaler.fit_transform(df_results_filt[['bdi']])

# Normalize BDI scores
scaler = MinMaxScaler()
df_results_filt['bdi_normalized'] = scaler.fit_transform(df_results_filt[['bdi']])
df_bdi = df_results_filt[['bdi_normalized', 'pid']].drop_duplicates(subset='pid', keep='first')

# Drop or include BDI in clustering
if include_bdi == False:
    df_results_filt.drop(columns=['bdi', 'bdi_normalized'], inplace=True)

# Assuming df_results is pre-loaded with the appropriate data
# Filter out the intercept and only include p-values < 0.05
alpha = 0.05
df_results_filt = df_results_filt[df_results_filt['variable'] != 'const']
df_results_filt = df_results_filt.groupby('pid').filter(lambda x: (x['permuted_p_value'] < alpha).any())

# Pivoting the DataFrame to get one row per participant with each coefficient as a column
df_pivot = df_results_filt.pivot(index='pid', columns='variable', values='partial_r2')

# Fill any missing values if necessary (depends on your data)
df_pivot.fillna(0, inplace=True)

# Normalize each participant's coefficients vector between 0 and 1
#df_pivot = df_pivot.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# Add normalized BDI score to the pivot
# First, ensure df_results_filt is indexed by 'pid' to align properly
if include_bdi == True:
    df_pivot['bdi_normalized'] = df_bdi.set_index('pid')['bdi_normalized']

# UMAP Dimensionality Reduction
reducer = umap.UMAP(n_neighbors=15, random_state=42)
umap_embedding = reducer.fit_transform(df_pivot)
#umap_embedding = df_pivot[['error', 'previous_feedback', 'trial']].values

# Define ranges for the parameters
min_cluster_sizes = range(2, 10)  # Adjust range according to your dataset size
min_samples_list = range(1, 10)   # Adjust range as needed
cluster_selection_epsilon_list = np.linspace(0.01, 2, 20)
best_score = -1
best_params = None

# Iterate over all combinations of min_cluster_size and min_samples
for min_cluster_size in tqdm(min_cluster_sizes, total=len(min_cluster_sizes)):
    for min_samples in min_samples_list:
        for cse in cluster_selection_epsilon_list:
            cse = float(cse)
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                        min_samples=min_samples,
                                        gen_min_span_tree=True,
                                        cluster_selection_epsilon=cse,
                                        )

            cluster_labels = clusterer.fit_predict(umap_embedding)

            # Calculate the Silhouette Score,
            # only if more than one cluster is found
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(umap_embedding, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_params = (min_cluster_size, min_samples, cse)



# HDBSCAN Clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=best_params[0],
                            gen_min_span_tree=True,
                            min_samples=best_params[1],
                            cluster_selection_epsilon=best_params[2],
                            )

cluster_labels = clusterer.fit_predict(umap_embedding)

# Prepare the plotting environment
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# UMAP Plot with cluster
scatter = ax[0].scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                        c=cluster_labels, cmap='brg', alpha=0.7)
ax[0].set_title('UMAP Projection of Participant partial_r2 by Cluster')
ax[0].set_xlabel('UMAP Dimension 1')
ax[0].set_ylabel('UMAP Dimension 2')
ax[0].spines[['top', 'right']].set_visible(False)
#colorbar = plt.colorbar(scatter, ax=ax[0])
#colorbar.set_label('Cluster Label')

# Coefficients Plot for each participant colored by cluster
# Create a color map for clusters
cluster_colors = plt.cm.brg(np.linspace(0, 1, len(set(cluster_labels))))
color_dict = {k: cluster_colors[i]
              for i, k
              in enumerate(sorted(set(cluster_labels)))}
legend_handles = []

for i, pid in enumerate(df_pivot.index):
    cluster_color = color_dict[cluster_labels[i]]
    label = f'Cluster {cluster_labels[i]}' if cluster_labels[i] != -1 else 'Outliers'
    # Only add the legend handle once per cluster
    if label not in [h.get_label() for h in legend_handles]:
        handle = plt.Line2D([], [], color=cluster_color, label=label)
        legend_handles.append(handle)
    ax[1].plot(df_pivot.columns, df_pivot.loc[pid], color=cluster_color)

ax[1].set_title('partial_r2 Values by Participant')
ax[1].set_xlabel('Variable')
ax[1].set_ylabel('partial_r2 Value')
ax[1].tick_params(axis='x', rotation=45)
ax[1].spines[['top', 'right']].set_visible(False)
ax[1].legend(handles=legend_handles, title='Participant Cluster',
             loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()


#%%
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt

# Assuming df_results is pre-loaded and includes 'bdi' scores
# Filter out the intercept and only include p-values < 0.05
alpha = 0.05
df_results_filt = df_results[df_results['variable'] != 'const']
df_results_filt = df_results_filt.groupby('pid').filter(lambda x:
                                                        (x['permuted_p_value']
                                                         < 0.05).any())

# Pivoting the DataFrame to get one row per participant with each coefficient as a column
df_pivot = df_results_filt.pivot(index='pid', columns='variable',
                                 values='partial_r2')

# Include bdi scores in the pivot, this time directly from df_results
df_bdi = df_results[['pid', 'bdi']].drop_duplicates().set_index('pid')
df_pivot = df_pivot.join(df_bdi)  # Join bdi scores to the coefficients

# Fill any missing values if necessary (depends on your data)
df_pivot.fillna(0, inplace=True)

# Normalize each participant's coefficients vector between 0 and 1 (excluding 'bdi')
df_pivot.loc[:, df_pivot.columns != 'bdi'] = df_pivot.loc[:,
                                                          df_pivot.columns
                                                          != 'bdi'].apply(
    lambda x: (x - x.min()) / (x.max() - x.min()), axis=1)

# UMAP Dimensionality Reduction
reducer = umap.UMAP(n_neighbors=25, random_state=42,)
umap_embedding = reducer.fit_transform(df_pivot.loc[:,
                                                    df_pivot.columns
                                                    != 'bdi'])


# Prepare the plotting environment
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# UMAP Plot colored by bdi
scatter = ax[0].scatter(umap_embedding[:, 0], umap_embedding[:, 1],
                        c=df_pivot['bdi'], cmap='Reds', alpha=0.7)
ax[0].set_title('UMAP Projection of Participant partial_r2 Colored by BDI')
ax[0].set_xlabel('UMAP Dimension 1')
ax[0].set_ylabel('UMAP Dimension 2')
ax[0].spines[['top', 'right']].set_visible(False)
plt.colorbar(scatter, ax=ax[0], label='BDI Score')

# Coefficients Plot for each participant colored by bdi
for i, pid in enumerate(df_pivot.index):
    ax[1].plot(df_pivot.columns[:-1],
               df_pivot.loc[pid, df_pivot.columns != 'bdi'],
               color=plt.cm.Reds(df_pivot.at[pid,
                                             'bdi'] / df_pivot['bdi'].max()))
ax[1].set_title('partial_r2 Values by Participant Colored by BDI')
ax[1].set_xlabel('Variable')
ax[1].set_ylabel('partial_r2 Value')
ax[1].tick_params(axis='x', rotation=45)
ax[1].spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%% df


df_temp = df_pivot.copy()
df_temp['cluster'] = cluster_labels
df_temp['pid'] = df_temp.index
df_temp.index = range(len(df_temp))

# merge cluster
# Assuming the 'cluster' column is confirmed to be in `temp_df`
# Perform the merge operation
new_df = df_a.merge(df_temp[['pid', 'cluster']], on='pid', how='left')


# only cluster participants
new_df_cluster = new_df[new_df['cluster']==0]


#%% Restructure df

# Get bdi score for each participant
bdi_scores = new_df_cluster.groupby('pid')['bdi'].first().reset_index()

# Get BDI
new_df_cluster['bdi'] = new_df_cluster.groupby('pid')['bdi'].transform('first')

# Compute the median 'bdi' score
median_bdi = new_df_cluster['bdi'].median()

# Create a new column 'bdi_level' with 'high' or 'low' based on the median split
new_df_cluster['bdi_level'] = ['high' if x > median_bdi
                               else 'low' for x in new_df_cluster['bdi']]

# Set up dataframe
df = new_df_cluster.copy()

# Create trial data where only single numbers
df['trial'] = [np.array(range(20)) for i in range(len(df))]
df['session'] = [[i]*20 for i in df['session'].values]
df['pid'] = [[i]*20 for i in df['pid'].values]
df['condition'] = [[i]*20 for i in df['condition_list'].values]
df['bdi'] = [[i]*20 for i in df['bdi'].values]
df['bdi_level'] = [[i]*20 for i in df['bdi_level'].values]
df['cluster'] = [[i]*20 for i in df['cluster'].values]

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
    'cluster': df['cluster'].values,
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
        'cluster': row['cluster'],
        #'feedback_sub_confidence': row['feedback_sub_confidence'],
    })

    # Append to the expanded DataFrame
    expanded_df = pd.concat([expanded_df, temp_df], ignore_index=True)


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
plt.xlabel('BDI', fontsize=16)
plt.ylabel('Metacognitive Bias', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14)
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

df['mean_confidence'] = df.groupby('pid')['confidence_task'].transform('mean')
df['feedback_aligment'] = df['mean_confidence'] - df['previous_feedback']
df['avg_feedback_aligment']  = df.groupby('pid')['feedback_aligment'].transform('mean')

#%% Correlate avg feedback aligment with BDI

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# Assuming 'df' is your DataFrame and it contains 'bdi' and 'metacogbias' columns
# df = pd.DataFrame({'bdi': [...], 'metacogbias': [...]})

# Calculate Pearson correlation and p-value
pearson_r, p_value = stats.pearsonr(df['bdi'], df['avg_feedback_aligment'])

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

# Annotate the plot with the Pearson correlation coefficient and p-value
#plt.annotate(f'r={pearson_r:.2f}, p={p_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
#             ha='left', va='top', fontsize=10, color='red')

# Add labels and legend
plt.xlabel('BDI', fontsize=16)
plt.ylabel('Mean confidence - feedback', fontsize=16)
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
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

# Annotate the plot with the Pearson correlation coefficient and p-value
#plt.annotate(f'r={pearson_r:.2f}, p={p_value:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
#             ha='left', va='top', fontsize=10, color='red')

# Add labels and legend
plt.xlabel('metacogbias')
plt.ylabel('Mean confidence - feedback')
plt.legend(fontsize=10)
axs.spines[['top', 'right']].set_visible(False)

# Show the plot
plt.show()

#%% assymetry

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
           # var_list.append(variance)

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
        plt.ylim(-0.05, 0.5)
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

