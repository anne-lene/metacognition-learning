# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:33:51 2024

@author: carll
"""
# Model recovery
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#%% Calculate confusion matrix

# Load data
#sim_data = pd.read_csv('simulated_conf_EXP2_data_more_deterministic.csv')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_best_fit_params.xlsx')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic.xlsx')
fit_data = pd.read_excel('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/variable_feedback/model_comparison/model_and_param_recovery/model_fits_EXP2_sim.xlsx')

# Define the order of simulated models
simulated_models = ['bias', 'win_stay', 'rw',
                    'rw_cond', 'ck', 'rwck',
                    'rwpd', 'rwp']

# Change to same naming convension as in sim_data
fit_data['simulated_model'] = fit_data['model']

# Melt fit_data to long format to handle all NLL columns as fitted models
nll_columns = [
    'nll_bias', 'nll_win_stay',
    'nll_rw_symm', 'nll_rw_cond', 'nll_ck',
    'nll_rwck', 'nll_rwpd', 'nll_rwp']

fit_data_long = fit_data.melt(
    id_vars=['pid', 'simulated_model'],
    value_vars=nll_columns,
    var_name='fitted_model',
    value_name='nll')

# Clean up fitted model names
fit_data_long['fitted_model'] = fit_data_long[
                                'fitted_model'].str.replace('nll_array_',
                                                            ''
                                              ).str.replace(
                                                            '_p',
                                                            ''
                                              ).str.replace(
                                                            'win_stay',
                                                            'wsls'
                                              ).str.replace(
                                                            'delta_rw',
                                                            'rwpd'
                                              ).str.replace(
                                                            'rw_symm',
                                                            'rw'
                                              ).str.replace(
                                                          'nll_',
                                                          '')

# Find the minimum NLL for each participant and simulated model
fit_data_long['min_nll'] = fit_data_long.groupby(['pid',
                                                  'simulated_model']
                                                 )['nll'].transform('min')

# Calculate the relative NLL
fit_data_long['relative_nll'] = fit_data_long['nll'] - fit_data_long['min_nll']

# Drop the min_nll column if you only need relative_nll
#fit_data_long.drop(columns='min_nll', inplace=True)

# Compute exp(-relative_nll)
fit_data_long['exp_nll'] = np.exp(-fit_data_long['relative_nll'])

# Normalize within each group to calculate probabilities
fit_data_long['prob_fit_given_sim'] = fit_data_long.groupby(
                                                     ['pid',
                                                     'simulated_model'
                                                     ]
                                                    )['exp_nll'
                                                     ].transform(lambda x:
                                                                 x / x.sum()
                                                                 )

# Drop exp_nll column if not needed
fit_data_long.drop(columns='exp_nll', inplace=True)

# Confusion matrix  - Simulated model on Y, Fitted model on X

# Pivot the data for the heatmap
heatmap_data = fit_data_long.pivot_table(
    index='simulated_model',
    columns='fitted_model',
    values='prob_fit_given_sim',
    aggfunc='mean'
)

# Sort the simulated and fitted models to match the intended order
model_order = ['bias', 'wsls', 'rw', 'rw_cond', 'ck', 'rwck', 'rwpd', 'rwp']
heatmap_data = heatmap_data.loc[model_order, model_order]

# Plot the Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    heatmap_data,
    annot=True,  # Show the NLL values in each cell
    fmt=".2f",   # Format the NLL values to 2 decimal places
    cmap="viridis",  # Choose a color map
    cbar_kws={'label':
              'p(fit model∣simulated model)'}  # Add a label for the color bar
)
plt.title("Confusion Matrix: p(fit model∣simulated model)")
plt.ylabel("Simulated Model")
plt.xlabel("Fitted Model")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%% Inversion matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
#sim_data = pd.read_csv('simulated_conf_EXP2_data_more_deterministic.csv')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_best_fit_params.xlsx')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic.xlsx')
#fit_data = pd.read_excel('scripts/old/data/sim_EXP2_model_metrics_sessions_CV_super_deterministic_testing123.xlsx')
fit_data = pd.read_excel('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/variable_feedback/model_comparison/model_and_param_recovery/model_fits_EXP2_sim.xlsx')

# Define the order of simulated models
simulated_models = ['bias', 'win_stay', 'rw',
                    'rw_cond', 'ck', 'rwck',
                    'rwpd', 'rwp']

# Assign the model names to a new column
fit_data['simulated_model'] = fit_data['model']

# Melt fit_data to long format to handle all NLL columns as fitted models
nll_columns = [
    'nll_bias', 'nll_win_stay', 'nll_rw_symm', 'nll_rw_cond',
    'nll_ck', 'nll_rwck', 'nll_rwpd', 'nll_rwp'
]

fit_data_long = fit_data.melt(
    id_vars=['pid', 'simulated_model'],
    value_vars=nll_columns,
    var_name='fitted_model',
    value_name='nll'
)

# Clean up fitted model names
fit_data_long['fitted_model'] = fit_data_long['fitted_model'].str.replace(
    'nll_', '').str.replace('win_stay', 'wsls').str.replace('rw_symm', 'rw')

# Find the minimum NLL for each participant and simulated model
fit_data_long['min_nll'] = fit_data_long.groupby(['pid', 'simulated_model'])['nll'].transform('min')

# Calculate the relative NLL
fit_data_long['relative_nll'] = fit_data_long['nll'] - fit_data_long['min_nll']

# Compute exp(-relative_nll)
fit_data_long['exp_nll'] = np.exp(-fit_data_long['relative_nll'])

# Normalize within each group to calculate p(fit | sim)
fit_data_long['prob_fit_given_sim'] = fit_data_long.groupby(['pid', 'simulated_model'])['exp_nll'].transform(lambda x: x / x.sum())

# Pivot the confusion matrix: p(fit | sim)
confusion_matrix = fit_data_long.pivot_table(
    index='simulated_model',
    columns='fitted_model',
    values='prob_fit_given_sim',
    aggfunc='mean'
)

# Normalize confusion matrix to compute p(sim | fit)
inversion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=0), axis=1)

# Sort the simulated and fitted models to match the intended order
model_order = ['bias', 'wsls', 'rw', 'rw_cond', 'ck', 'rwck', 'rwpd', 'rwp']
inversion_matrix = inversion_matrix.loc[model_order, model_order]

# Plot the Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    inversion_matrix,
    annot=True,  # Show the probabilities in each cell
    fmt=".2f",   # Format the values to 2 decimal places
    cmap="viridis",  # Choose a color map
    cbar_kws={'label': 'p(simulated model | fitted model)'}  # Add a label for the color bar
)

plt.title("Inversion Matrix: p(simulated model | fitted model)")
plt.ylabel("Simulated Model")
plt.xlabel("Fitted Model")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%% Fewer models: Confusion matrix

# Load data
#sim_data = pd.read_csv('simulated_conf_EXP2_data_more_deterministic.csv')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_best_fit_params.xlsx')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic.xlsx')
#fit_data = pd.read_excel('scripts/old/data/sim_EXP2_model_metrics_sessions_CV_super_deterministic_testing123.xlsx')
fit_data = pd.read_excel('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/variable_feedback/model_comparison/model_and_param_recovery/model_fits_EXP2_sim.xlsx')

# Define the order of simulated models
simulated_models = [#'bias',
                    'win_stay',
                   # 'rw',
                    'rw_cond'
                    'ck',
                   # 'rwck'
                   # 'rwpd',
                    'rwp']

# Change to same naming convension as in sim_data
fit_data['simulated_model'] = fit_data['model']

# Melt fit_data to long format to handle all NLL columns as fitted models
nll_columns = [
   # 'nll_bias',
    'nll_win_stay',
   # 'nll_rw_symm',
    'nll_rw_cond',
    'nll_ck',
   # 'nll_rwck',
   # 'nll_rwpd',
    'nll_rwp'
   ]

fit_data_long = fit_data.melt(
    id_vars=['pid', 'simulated_model'],
    value_vars=nll_columns,
    var_name='fitted_model',
    value_name='nll')

# Clean up fitted model names
fit_data_long['fitted_model'] = fit_data_long[
                                'fitted_model'].str.replace('nll_array_',
                                                            ''
                                              ).str.replace(
                                                            '_p',
                                                            ''
                                              ).str.replace(
                                                            'win_stay',
                                                            'wsls'
                                              ).str.replace(
                                                            'delta_rw',
                                                            'rwpd'
                                              ).str.replace(
                                                            'rw_symm',
                                                            'rw'
                                              ).str.replace(
                                                          'nll_',
                                                          '')

# Find the minimum NLL for each participant and simulated model
fit_data_long['min_nll'] = fit_data_long.groupby(['pid',
                                                  'simulated_model']
                                                 )['nll'].transform('min')

# Calculate the relative NLL
fit_data_long['relative_nll'] = fit_data_long['nll'] - fit_data_long['min_nll']

# Drop the min_nll column if you only need relative_nll
#fit_data_long.drop(columns='min_nll', inplace=True)

# Compute exp(-relative_nll)
fit_data_long['exp_nll'] = np.exp(-fit_data_long['relative_nll'])

# Normalize within each group to calculate probabilities
fit_data_long['prob_fit_given_sim'] = fit_data_long.groupby(
                                                     ['pid',
                                                     'simulated_model'
                                                     ]
                                                    )['exp_nll'
                                                     ].transform(lambda x:
                                                                 x / x.sum()
                                                                 )

# Drop exp_nll column if not needed
fit_data_long.drop(columns='exp_nll', inplace=True)

# Confusion matrix  - Simulated model on Y, Fitted model on X

# Pivot the data for the heatmap
heatmap_data = fit_data_long.pivot_table(
    index='simulated_model',
    columns='fitted_model',
    values='prob_fit_given_sim',
    aggfunc='mean'
)

# Sort the simulated and fitted models to match the intended order
model_order = [
              # 'bias',
                'wsls',
              # 'rw',
                'rw_cond',
                'ck',
              # 'rwck',
              # 'rwpd',
                'rwp'
              ]
heatmap_data = heatmap_data.loc[model_order, model_order]

# Plot the Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    heatmap_data,
    annot=True,  # Show the NLL values in each cell
    fmt=".2f",   # Format the NLL values to 2 decimal places
    cmap="viridis",  # Choose a color map
    cbar_kws={'label':
              'p(fit model∣simulated model)'}  # Add a label for the color bar
)
plt.title("Confusion Matrix: p(fit model∣simulated model)")
plt.ylabel("Simulated Model")
plt.xlabel("Fitted Model")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

#%% Fewer models: Inversoin matrix

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_best_fit_params.xlsx')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic.xlsx')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_more_deterministic.xlsx')
#fit_data = pd.read_excel('scripts/old/data/sim_EXP2_model_metrics_sessions_CV_super_deterministic_testing123.xlsx')
fit_data = pd.read_excel('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/variable_feedback/model_comparison/model_and_param_recovery/model_fits_EXP2_sim.xlsx')

# Define the order of models to consider
model_order = ['wsls', 'rw_cond', 'ck', 'rwp']

# Assign the model names to a new column
fit_data['simulated_model'] = fit_data['model']

# Melt fit_data to long format to handle all NLL columns as fitted models
nll_columns = [
   'nll_win_stay', 'nll_rw_cond', 'nll_ck', 'nll_rwp'
]

fit_data_long = fit_data.melt(
    id_vars=['pid', 'simulated_model'],
    value_vars=nll_columns,
    var_name='fitted_model',
    value_name='nll'
)

# Clean up fitted model names
fit_data_long['fitted_model'] = fit_data_long['fitted_model'].str.replace(
    'nll_', '').str.replace('win_stay', 'wsls').str.replace('rw_symm', 'rw')

# Filter to only include the selected models
fit_data_long = fit_data_long[fit_data_long['fitted_model'].isin(model_order)]
fit_data_long = fit_data_long[fit_data_long['simulated_model'].isin(model_order)]

# Find the minimum NLL for each participant and simulated model
fit_data_long['min_nll'] = fit_data_long.groupby(['pid', 'simulated_model'])['nll'].transform('min')

# Calculate the relative NLL
fit_data_long['relative_nll'] = fit_data_long['nll'] - fit_data_long['min_nll']

# Compute exp(-relative_nll)
fit_data_long['exp_nll'] = np.exp(-fit_data_long['relative_nll'])

# Normalize within each group to calculate p(fit | sim)
fit_data_long['prob_fit_given_sim'] = fit_data_long.groupby(['pid', 'simulated_model'])['exp_nll'].transform(lambda x: x / x.sum())

# Pivot the confusion matrix: p(fit | sim)
confusion_matrix = fit_data_long.pivot_table(
    index='simulated_model',
    columns='fitted_model',
    values='prob_fit_given_sim',
    aggfunc='mean'
)

# Normalize confusion matrix to compute p(sim | fit)
inversion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=0), axis=1)

# Sort the simulated and fitted models to match the intended order
inversion_matrix = inversion_matrix.loc[model_order, model_order]

# Plot the Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(
    inversion_matrix,
    annot=True,  # Show the probabilities in each cell
    fmt=".2f",   # Format the values to 2 decimal places
    cmap="viridis",  # Choose a color map
    cbar_kws={'label': 'p(simulated model | fitted model)'}  # Add a label for the color bar
)
plt.title("Inversion Matrix: p(simulated model | fitted model)")
plt.ylabel("Simulated Model")
plt.xlabel("Fitted Model")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =============================================================================
# #%% Baysian Inversion matrix
#
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Load data
# #fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic.xlsx')
# fit_data = pd.read_excel('scripts/old/data/sim_EXP2_model_metrics_sessions_CV_super_deterministic_testing123.xlsx')
#
# # Define the order of models to consider
# model_order = ['bias', 'wsls', 'rw', 'rw_cond', 'ck', 'rwck', 'rwpd']
#
# # Assign the model names to a new column
# fit_data['simulated_model'] = fit_data['model']
#
# # Melt fit_data to long format to handle all NLL columns as fitted models
# nll_columns = [
#     'nll_bias', 'nll_win_stay', 'nll_rw_symm',  'nll_rw_cond', 'nll_ck', 'nll_rwck', 'nll_rwpd'
# ]
#
#
# fit_data_long = fit_data.melt(
#     id_vars=['pid', 'simulated_model'],
#     value_vars=nll_columns,
#     var_name='fitted_model',
#     value_name='nll'
# )
#
# # Clean up fitted model names
# fit_data_long['fitted_model'] = fit_data_long['fitted_model'].str.replace(
#     'nll_', '').str.replace('win_stay', 'wsls').str.replace('rw_symm', 'rw')
#
# # Filter to only include the selected models
# fit_data_long = fit_data_long[fit_data_long['fitted_model'].isin(model_order)]
# fit_data_long = fit_data_long[fit_data_long['simulated_model'].isin(model_order)]
#
# # Find the minimum NLL for each participant and simulated model
# fit_data_long['min_nll'] = fit_data_long.groupby(['pid', 'simulated_model'])['nll'].transform('min')
#
# # Calculate the relative NLL
# fit_data_long['relative_nll'] = fit_data_long['nll'] - fit_data_long['min_nll']
#
# # Compute exp(-relative_nll)
# fit_data_long['exp_nll'] = np.exp(-fit_data_long['relative_nll'])
#
# # Normalize within each group to calculate p(fit | sim)
# fit_data_long['prob_fit_given_sim'] = fit_data_long.groupby(['pid', 'simulated_model'])['exp_nll'].transform(lambda x: x / x.sum())
#
# # Pivot the confusion matrix: p(fit | sim)
# confusion_matrix = fit_data_long.pivot_table(
#     index='simulated_model',
#     columns='fitted_model',
#     values='prob_fit_given_sim',
#     aggfunc='mean'
# )
#
# # Compute priors on simulated models (uniform prior)
# num_simulated_models = len(model_order)
# prior_sim = 1 / num_simulated_models
#
# # Apply Bayes' Rule to compute p(sim | fit)
# bayesian_inversion_matrix = confusion_matrix * prior_sim
# bayesian_inversion_matrix = bayesian_inversion_matrix.div(bayesian_inversion_matrix.sum(axis=0), axis=1)
#
# # Sort the simulated and fitted models to match the intended order
# bayesian_inversion_matrix = bayesian_inversion_matrix.loc[model_order, model_order]
#
# # Plot the Heatmap
# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     bayesian_inversion_matrix,
#     annot=True,  # Show the probabilities in each cell
#     fmt=".2f",   # Format the values to 2 decimal places
#     cmap="viridis",  # Choose a color map
#     cbar_kws={'label': 'p(simulated model | fitted model)'}  # Add a label for the color bar
# )
# plt.title("Bayesian Inversion Matrix: p(simulated model | fitted model)")
# plt.ylabel("Simulated Model")
# plt.xlabel("Fitted Model")
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
#
# #%% Compute global probabilities based on the average nll for
# #   each simulated vs fitted model combination
#
# aggregated_nll = fit_data_long.groupby(['simulated_model',
#                                         'fitted_model'])['nll'].mean()
# aggregated_nll = aggregated_nll.reset_index()
#
# # Compute exp(-relative_nll) and normalize
# aggregated_nll['exp_nll'] = np.exp(-aggregated_nll['nll'])
# aggregated_nll['prob_fit_given_sim'] = aggregated_nll.groupby(
#                                                             'simulated_model'
#                                                               )['exp_nll'
#                                                                 ].transform(
#                                                                   lambda x:
#                                                                   x / x.sum()
#                                                                   )
#
# # Confusion matrix  - Simulated model on Y, Fitted model on X
#
# # Pivot the data for the heatmap
# heatmap_data = aggregated_nll.pivot_table(
#     index='simulated_model',
#     columns='fitted_model',
#     values='prob_fit_given_sim',
#     aggfunc='mean'
# )
#
# # Sort the simulated and fitted models to match the intended order
# model_order = ['bias', 'wsls', 'rw', 'rw_cond', 'ck', 'rwck', 'rwpd']
# heatmap_data = heatmap_data.loc[model_order, model_order]
#
# # Plot the Heatmap
# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     heatmap_data,
#     annot=True,  # Show the NLL values in each cell
#     fmt=".2f",   # Format the NLL values to 2 decimal places
#     cmap="viridis",  # Choose a color map
#     cbar_kws={'label':
#               'p(fit model∣simulated model)'}  # Add a label for the color bar
# )
# plt.title("Confusion Matrix: p(fit model∣simulated model)")
# plt.ylabel("Simulated Model")
# plt.xlabel("Fitted Model")
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
#
#
# #%%
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # Load data
# fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_more_deterministic.xlsx')
#
# # Assign the simulated model names to a new column
# g = fit_data.groupby('pid')
# fit_data['simulated_model'] = fit_data['model']
#
# # Standardize column names in fit_data
# fit_data.rename(columns={
#     'nll_win_stay': 'nll_wsls',
#     'nll_rw_symm': 'nll_rw'
# }, inplace=True)
#
# # Check if all required columns are present
# nll_columns = [
#     'nll_bias', 'nll_wsls', 'nll_rw',
#     'nll_rw_cond', 'nll_ck', 'nll_rwck', 'nll_rwpd'
# ]
# missing_columns = [col for col in nll_columns if col not in fit_data.columns]
# if missing_columns:
#     raise ValueError(f"Missing columns in fit_data: {missing_columns}")
#
# # Melt the data to long format
# fit_data_long = fit_data.melt(
#     id_vars=['pid', 'simulated_model'],
#     value_vars=nll_columns,
#     var_name='fitted_model',
#     value_name='nll'
# )
#
#
# # Clean up fitted model names
# fit_data_long['fitted_model'] = fit_data_long['fitted_model'].str.replace(
#     'nll_array_', '').str.replace('_p', '').replace(
#     {'win_stay': 'wsls', 'delta_rw': 'rwpd', 'rw_symm': 'rw'})
#
# # Identify the best (lowest NLL) fitted model for each simulation
# fit_data_long['best_fit'] = fit_data_long.groupby(['pid',
#                                                    'simulated_model'
#                                                    ])['nll'].transform(
#                                                    'min') == fit_data_long[
#                                                    'nll']
#
# # Count how often each fitted model was the best for each simulated model
# confusion_counts = fit_data_long[fit_data_long['best_fit']].groupby(
#     ['simulated_model', 'fitted_model']).size().unstack(fill_value=0)
#
# # Normalize counts to probabilities (percentages)
# confusion_matrix = (confusion_counts.T / confusion_counts.sum(axis=1)).T
#
# # Mapping from short names to full names
# name_mapping = {
#     'bias': 'nll_bias',
#     'wsls': 'nll_wsls',
#     'rw': 'nll_rw',
#     'rw_cond': 'nll_rw_cond',
#     'ck': 'nll_ck',
#     'rwck': 'nll_rwck',
#     'rwpd': 'nll_rwpd'
# }
#
# # Update the confusion matrix index using the mapping
# confusion_matrix.rename(index=name_mapping, inplace=True)
#
# # Reorder the rows and columns to match the model_order
# model_order = ['nll_bias', 'nll_wsls', 'nll_rw', 'nll_rw_cond',
#                'nll_ck', 'nll_rwck', 'nll_rwpd']
# confusion_matrix = confusion_matrix.reindex(index=model_order,
#                                             columns=model_order,
#                                             fill_value=0)
#
# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     confusion_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="viridis",
#     cbar_kws={'label': 'Probability (%)'}
# )
# plt.title("Confusion Matrix: Probability of Best Fitted Model")
# plt.ylabel("Simulated Model")
# plt.xlabel("Fitted Model")
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)
# plt.tight_layout()
# plt.show()
#
# #%%
#
# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # File path check
# file_path = 'sim_EXP2_model_metrics_sessions_CV_more_deterministic.xlsx'
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found: {file_path}")
#
# # Load data
# fit_data = pd.read_excel(file_path)
#
# # Standardize column names in fit_data
# fit_data.rename(columns={
#     'nll_win_stay': 'nll_wsls',
#     'nll_rw_symm': 'nll_rw'
# }, inplace=True)
#
# # Define NLL columns (after renaming)
# nll_columns = [
#     'nll_bias', 'nll_wsls', 'nll_rw',
#     'nll_rw_cond', 'nll_ck', 'nll_rwck', 'nll_rwpd'
# ]
#
# # Check if all required columns are present
# missing_columns = [col for col in nll_columns if col not in fit_data.columns]
# if missing_columns:
#     raise ValueError(f"Missing columns in fit_data: {missing_columns}")
#
# # Assign the simulated model names to a new column
# fit_data['simulated_model'] = fit_data['model'].replace({
#     'win_stay': 'wsls',
#     'delta_p_rw': 'rwpd'
# })
#
# # Melt fit_data to long format for NLL comparison
# fit_data_long = fit_data.melt(
#     id_vars=['pid', 'simulated_model'],
#     value_vars=nll_columns,
#     var_name='fitted_model',
#     value_name='nll'
# )
#
# # Clean up fitted model names
# fit_data_long['fitted_model'] = fit_data_long['fitted_model'].replace({
#     'nll_wsls': 'wsls',
#     'nll_rw': 'rw',
#     'nll_rw_cond': 'rw_cond',
#     'nll_ck': 'ck',
#     'nll_rwck': 'rwck',
#     'nll_rwpd': 'rwpd'
# })
#
# # Identify the best (lowest NLL) fitted model for each simulation
# fit_data_long['best_fit'] = fit_data_long.groupby(['pid', 'simulated_model'])['nll'].transform('min') == fit_data_long['nll']
#
# # Count how often each fitted model was the best for each simulated model
# confusion_counts = fit_data_long[fit_data_long['best_fit']].groupby(
#     ['simulated_model', 'fitted_model']).size().unstack(fill_value=0)
#
# # Normalize counts to probabilities (percentages)
# confusion_matrix = (confusion_counts.T / confusion_counts.sum(axis=1)).T
#
# # Mapping from short names to full names
# name_mapping = {
#     'bias': 'nll_bias',
#     'wsls': 'nll_wsls',
#     'rw': 'nll_rw',
#     'rw_cond': 'nll_rw_cond',
#     'ck': 'nll_ck',
#     'rwck': 'nll_rwck',
#     'rwpd': 'nll_rwpd'
# }
#
# # Update the confusion matrix index using the mapping
# confusion_matrix.rename(index=name_mapping, inplace=True)
#
# # Reorder the rows and columns to match the model_order
# model_order = ['nll_bias', 'nll_wsls', 'nll_rw', 'nll_rw_cond',
#                'nll_ck', 'nll_rwck', 'nll_rwpd']
# confusion_matrix = confusion_matrix.reindex(index=model_order,
#                                             columns=model_order,
#                                             fill_value=0)
#
# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(
#     confusion_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="viridis",
#     cbar_kws={'label': 'Probability (%)'}
# )
# plt.title("Confusion Matrix: Probability of Best Fitted Model")
# plt.ylabel("Simulated Model")
# plt.xlabel("Fitted Model")
# plt.xticks(rotation=45, ha='right', fontsize=10)
# plt.yticks(fontsize=10)
# plt.tight_layout()
# plt.show()
#
# #%%
# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # File path check
# #file_path = 'sim_EXP2_model_metrics_sessions_CV_more_deterministic.xlsx'
# file_path = 'sim_EXP2_model_metrics_sessions_CV_best_fit_params.xlsx'
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found: {file_path}")
#
# # Load data
# fit_data = pd.read_excel(file_path)
#
# # Assign the simulated model names to a new column
# fit_data['simulated_model'] = fit_data['model'].replace({
#     'win_stay': 'wsls',
#     'delta_p_rw': 'rwpd'
# })
#
# # Standardize column names in fit_data
# fit_data.rename(columns={
#     'nll_win_stay': 'nll_wsls',
#     'nll_rw_symm': 'nll_rw'
# }, inplace=True)
#
# # Remove 'nll_' from column names
# fit_data.columns = fit_data.columns.str.replace('nll_', '', regex=False)
#
# # Check if all required columns are present
# nll_columns = [
#     'bias', 'wsls', 'rw',
#     'rw_cond', 'ck', 'rwck', 'rwpd'
# ]
#
# missing_columns = [col for col in nll_columns if col not in fit_data.columns]
# if missing_columns:
#     raise ValueError(f"The following required columns are missing from fit_data: {missing_columns}. "
#                      f"Ensure the column names match the expected names in the dataset.")
#
# # Melt the data to long format
# fit_data_long = fit_data.melt(
#     id_vars=['pid', 'simulated_model'],
#     value_vars=nll_columns,
#     var_name='fitted_model',
#     value_name='nll'
# )
#
# # Clean up fitted model names
# fit_data_long['fitted_model'] = fit_data_long['fitted_model'].replace({
#     'nll_array_': '', '_p': ''
# }, regex=True).replace({
#     'win_stay': 'wsls',
#     'delta_rw': 'rwpd',
#     'rw_symm': 'rw'
# })
#
#
# # Identify the best (lowest NLL) fitted model for each simulation
# group = fit_data_long.groupby(['pid', 'simulated_model'])
# fit_data_long['best_fit']  = group['nll'].transform('min') == fit_data_long['nll']
#
# # Count how often each fitted model was the best for each simulated model
# confusion_counts = fit_data_long[fit_data_long['best_fit']].groupby(
#     ['simulated_model', 'fitted_model']).size().unstack(fill_value=0)
#
# # Normalize counts to probabilities (percentages)
# confusion_matrix = confusion_counts.div(confusion_counts.sum(axis=1), axis=0)
#
# # Remove 'nll_' from index and column names in the confusion matrix
# confusion_matrix.rename(columns=lambda x: x.replace('nll_', ''),
#                         index=lambda x: x.replace('nll_', ''), inplace=True)
#
# # Reorder the rows and columns to match the model_order
# model_order = ['bias', 'wsls', 'rw', 'rw_cond', 'ck', 'rwck', 'rwpd']
# confusion_matrix = confusion_matrix.reindex(index=model_order,
#                                             columns=model_order, fill_value=0)
#
# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(
#     confusion_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="viridis",
#     cbar_kws={'label': 'Probability (%)'},
#     annot_kws={"size": 12}
#     )
#
# # Set font size for the colorbar label
# cbar = ax.collections[0].colorbar
# cbar.ax.set_ylabel('Probability (%)', fontsize=14)
# cbar.ax.yaxis.set_tick_params(labelsize=14)
#
# plt.title("Confusion Matrix: Probability of Best Fitted Model",
#           fontsize=14)
# plt.ylabel("Simulated Model", fontsize=14)
# plt.xlabel("Fitted Model", fontsize=14)
# plt.xticks(rotation=45, ha='right', fontsize=14)
# plt.yticks(rotation=360, fontsize=14)
# plt.tight_layout()
# plt.show()
#
# #%% Reacreating model recovery from Anne's thesis: RW vs RW_cond
# import os
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # File path check
# file_path = 'sim_EXP2_model_metrics_sessions_CV_more_deterministic.xlsx'
# if not os.path.exists(file_path):
#     raise FileNotFoundError(f"File not found: {file_path}")
#
# # Load data
# fit_data = pd.read_excel(file_path)
#
# # Assign the simulated model names to a new column
# fit_data['simulated_model'] = fit_data['model'].replace({
#     'win_stay': 'wsls',
#     'delta_p_rw': 'rwpd'
# })
#
# # Standardize column names in fit_data
# fit_data.rename(columns={
#     'nll_win_stay': 'nll_wsls',
#     'nll_rw_symm': 'nll_rw'
# }, inplace=True)
#
# # Remove 'nll_' from column names
# fit_data.columns = fit_data.columns.str.replace('nll_', '', regex=False)
#
# # Check if required columns are present
# nll_columns = ['rw', 'rw_cond']
#
# missing_columns = [col for col in nll_columns if col not in fit_data.columns]
# if missing_columns:
#     raise ValueError(f"The following required columns are missing from fit_data: {missing_columns}. "
#                      f"Ensure the column names match the expected names in the dataset.")
#
# # Filter data to include only RW and RW_cond models
# fit_data_long = fit_data.melt(
#     id_vars=['pid', 'simulated_model'],
#     value_vars=nll_columns,
#     var_name='fitted_model',
#     value_name='nll'
# )
#
# # Identify the best (lowest NLL) fitted model for each simulation
# group = fit_data_long.groupby(['pid', 'simulated_model'])
# fit_data_long['best_fit'] = group['nll'].transform('min') == fit_data_long['nll']
#
# # Count how often each fitted model was the best for each simulated model
# confusion_counts = fit_data_long[fit_data_long['best_fit']].groupby(
#     ['simulated_model', 'fitted_model']).size().unstack(fill_value=0)
#
# # Normalize counts to probabilities (percentages)
# confusion_matrix = confusion_counts.div(confusion_counts.sum(axis=1), axis=0)
#
# # Remove 'nll_' from index and column names in the confusion matrix
# confusion_matrix.rename(columns=lambda x: x.replace('nll_', ''),
#                         index=lambda x: x.replace('nll_', ''), inplace=True)
#
# # Reorder the rows and columns to match the model_order
# model_order = ['rw', 'rw_cond']
# confusion_matrix = confusion_matrix.reindex(index=model_order,
#                                             columns=model_order, fill_value=0)
#
# # Plot the heatmap
# plt.figure(figsize=(8, 6))
# ax = sns.heatmap(
#     confusion_matrix,
#     annot=True,
#     fmt=".2f",
#     cmap="viridis",
#     cbar_kws={'label': 'Probability (%)'},
#     annot_kws={"size": 12}
#     )
#
# # Set font size for the colorbar label
# cbar = ax.collections[0].colorbar
# cbar.ax.set_ylabel('Probability (%)', fontsize=14)
# cbar.ax.yaxis.set_tick_params(labelsize=14)
#
# plt.title("Confusion Matrix: Probability of Best Fitted Model",
#           fontsize=14)
# plt.ylabel("Simulated Model", fontsize=14)
# plt.xlabel("Fitted Model", fontsize=14)
# plt.xticks(rotation=45, ha='right', fontsize=14)
# plt.yticks(rotation=360, fontsize=14)
# plt.tight_layout()
# plt.show()
# =============================================================================
