# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:42:55 2025

@author: carll
"""
# Parameter Recovery

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Load data
sim_data = pd.read_csv('simulated_conf_EXP2_data_more_deterministic.csv')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_more_deterministic.xlsx')

#sim_data = pd.read_csv('simulated_conf_EXP2_data_best_fit_params.csv')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_best_fit_params.xlsx')

#sim_data = pd.read_csv('simulated_conf_EXP2_data_super_deterministic.csv')
#fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic.xlsx')
fit_data = pd.read_excel('sim_EXP2_model_metrics_sessions_CV_super_deterministic_testing123.xlsx')


# Define the order of simulated models
simulated_models = ['bias', 'win_stay', 'rw',
                    'rw_cond', 'ck', 'rwck',
                    'delta_p_rw']

# Change to same naming convension as in sim_data
fit_data['model'] = fit_data['model'].replace('win_stay',
                                              'wsls'
                                              ).replace('delta_p_rw',
                                                        'rwpd')

# Parameters mapping
params = ['bias_mean', 'bias_sigma', 'wsls_sigma', 'wsls_win_boundary',
          'rw_alpha', 'rw_sigma', 'rw_cond_alpha_neut', 'rw_cond_alpha_pos',
          'rw_cond_alpha_neg', 'rw_cond_sigma', 'ck_alpha', 'ck_sigma',
          'ck_beta', 'rwck_alpha', 'rwck_alpha_ck', 'rwck_sigma',
          'rwck_sigma_ck', 'rwck_beta', 'rwck_beta_ck', 'rwpd_alpha',
          'rwpd_sigma', 'rwpd_w_rw', 'rwpd_w_delta_p']

fit_data_params = ['mean_bias', 'sd_bias',
                   'std_WSLS','win_boundary_WSLS',
                   'alpha_rw_symm', 'sigma_rw_symm',
                   'alpha_neut_rw_cond', 'alphaos_rw_cond',
                   'alpha_neg_rw_cond', 'sigma_rw_cond',
                   'alpha_ck', 'sigma_ck', 'beta_ck',
                   'alpha_rwck', 'alpha_ck_rwck',
                   'sigma_rwck', 'sigma_ck_rwck',
                   'beta_rwck', 'ck_beta_rwck',
                   'alpha_rwpd', 'sigma_rwpd',
                   'w_rw_rwpd', 'w_performance_rwpd']

# Group simulated data by pid and model, taking the first occurrence of each parameter
summary_sim_data = sim_data.groupby(['pid', 'model'])[params].first().reset_index()

# Merge simulated and fitted data
merged_data = pd.merge(summary_sim_data, fit_data, on=['pid', 'model'], suffixes=('_sim', '_fit'))

# Define the parameter mappings
sim_params = ['bias_mean', 'bias_sigma',
              'wsls_sigma', 'wsls_win_boundary',
              'rw_alpha', 'rw_sigma',
              'rw_cond_alpha_neut', 'rw_cond_alpha_pos', 'rw_cond_alpha_neg', 'rw_cond_sigma',
              'ck_alpha', 'ck_sigma', 'ck_beta',
              'rwck_alpha', 'rwck_alpha_ck', 'rwck_sigma', 'rwck_sigma_ck', 'rwck_beta', 'rwck_beta_ck',
              'rwpd_alpha','rwpd_sigma', 'rwpd_w_rw', 'rwpd_w_delta_p']

fit_params = ['mean_bias', 'sd_bias',
              'std_WSLS','win_boundary_WSLS',
              'alpha_rw_symm', 'sigma_rw_symm',
              'alpha_neut_rw_cond', 'alphaos_rw_cond', 'alpha_neg_rw_cond', 'sigma_rw_cond',
              'alpha_ck', 'sigma_ck', 'beta_ck',
              'alpha_rwck', 'alpha_ck_rwck','sigma_rwck', 'sigma_ck_rwck', 'beta_rwck', 'ck_beta_rwck',
              'alpha_rwpd', 'sigma_rwpd', 'w_rw_rwpd', 'w_performance_rwpd']

results = []
sim_params = [ 'rw_cond_alpha_neut', 'rw_cond_alpha_pos', 'rw_cond_alpha_neg']
fit_params = ['alpha_neut_rw_cond', 'alphaos_rw_cond', 'alpha_neg_rw_cond']

#sim_params = ['rw_alpha', 'rw_sigma']
#fit_params = ['alpha_rw_symm', 'sigma_rw_symm']

data = merged_data[merged_data.model == 'rw_cond']

for sim_param, fit_param in zip(sim_params, fit_params):
    sim_values = data[sim_param].values.reshape(-1, 1)
    fit_values = data[fit_param].values

    # Calculate recovery metrics
    correlation, p_value = pearsonr(sim_values.flatten(), fit_values)
    mae = mean_absolute_error(sim_values, fit_values)
    rmse = np.sqrt(mean_squared_error(sim_values, fit_values))

    # Fit linear regression for the trendline
    reg = LinearRegression().fit(sim_values, fit_values)
    trendline = reg.predict(sim_values)

    # Store results
    results.append({'Parameter': sim_param, 'Correlation': correlation, 'MAE': mae, 'RMSE': rmse})

    # Plot recovery with trendline
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    plt.scatter(sim_values, fit_values, alpha=0.6, label='Data')
    plt.plot(sim_values, trendline, 'r--', label='Trendline')
    plt.plot([sim_values.min(), sim_values.max()], [sim_values.min(), sim_values.max()], 'k-', label='Identity Line')
    plt.xlabel('Simulated')
    plt.ylabel('Recovered')
    plt.title(f'Parameter Recovery: {sim_param} (r={correlation:.2f}, p={p_value:.3f})')
    plt.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.show()

#% Vizualise dataframe
print("Parameter Recovery Results:")
print(pd.DataFrame(results))

#%% color sigma
sim_params = ['rw_alpha', 'rw_sigma']
fit_params = ['alpha_rw_symm', 'sigma_rw_symm']
data = merged_data[merged_data.model == 'rw']

results = []

for sim_param, fit_param in zip(sim_params, fit_params):
    sim_values = data[sim_param].values.reshape(-1, 1)
    fit_values = data[fit_param].values
    sigma_values = data['rw_sigma'].values  # Extract sigma values for coloring

    # Calculate recovery metrics
    correlation, _ = pearsonr(sim_values.flatten(), fit_values)
    mae = mean_absolute_error(sim_values, fit_values)
    rmse = np.sqrt(mean_squared_error(sim_values, fit_values))

    # Fit linear regression for the trendline
    reg = LinearRegression().fit(sim_values, fit_values)
    trendline = reg.predict(sim_values)

    # Store results
    results.append({'Parameter': sim_param, 'Correlation': correlation, 'MAE': mae, 'RMSE': rmse})

    # Plot recovery with trendline, colored by sigma values
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    scatter = plt.scatter(sim_values.flatten(), fit_values, c=sigma_values, cmap='viridis', alpha=0.6, label='Data')
    plt.colorbar(scatter, label='Sigma Value')
    plt.plot(sim_values, trendline, 'r--', label='Trendline')
    plt.plot([sim_values.min(), sim_values.max()], [sim_values.min(), sim_values.max()], 'k-', label='Identity Line')
    plt.xlabel('Simulated')
    plt.ylabel('Recovered')
    plt.title(f'Parameter Recovery: {sim_param} (r={correlation:.2f})')
    plt.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.show()

#%% remove sigma < 20

# Filter data to include only rows where sigma is 20 or higher
filtered_data = data[data['rw_sigma'] <= 20]

# Reset the results list
results = []

# Loop through parameters for recovery analysis
for sim_param, fit_param in zip(sim_params, fit_params):
    sim_values = filtered_data[sim_param].values.reshape(-1, 1)
    fit_values = filtered_data[fit_param].values
    sigma_values = filtered_data['rw_sigma'].values  # Extract sigma values for coloring

    # Calculate recovery metrics
    correlation, _ = pearsonr(sim_values.flatten(), fit_values)
    mae = mean_absolute_error(sim_values, fit_values)
    rmse = np.sqrt(mean_squared_error(sim_values, fit_values))

    # Fit linear regression for the trendline
    reg = LinearRegression().fit(sim_values, fit_values)
    trendline = reg.predict(sim_values)

    # Store results
    results.append({'Parameter': sim_param, 'Correlation': correlation, 'MAE': mae, 'RMSE': rmse})

    # Plot recovery with trendline, colored by sigma values
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    scatter = plt.scatter(sim_values.flatten(), fit_values, c=sigma_values, cmap='viridis', alpha=0.6, label='Data')
    plt.colorbar(scatter, label='Sigma Value')
    plt.plot(sim_values, trendline, 'r--', label='Trendline')
    plt.plot([sim_values.min(), sim_values.max()], [sim_values.min(), sim_values.max()], 'k-', label='Identity Line')
    plt.xlabel('Simulated')
    plt.ylabel('Recovered')
    plt.title(f'Parameter Recovery (Sigma >= 20): {sim_param} (r={correlation:.2f})')
    plt.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.show()



#%%
# =============================================================================
#
# from sklearn.linear_model import LinearRegression
# from scipy.stats import pearsonr
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# import matplotlib.pyplot as plt
# import numpy as np
#
# # Sample `sim_params`, `fit_params`, and `data` setup
# sim_params = ['rw_alpha', 'rw_sigma']
# fit_params = ['alpha_rw_symm_p', 'sigma_rw_symm_p']
# results = []
#
# # Assuming `data` is a pre-existing DataFrame
# # Add a sigma color column for plotting
# sigma_values = data['rw_sigma']
#
# for sim_param, fit_param in zip(sim_params, fit_params):
#     sim_values = data[sim_param].values.reshape(-1, 1)
#     fit_values = data[fit_param].values
#
#     # Calculate recovery metrics
#     correlation, _ = pearsonr(sim_values.flatten(), fit_values)
#     mae = mean_absolute_error(sim_values, fit_values)
#     rmse = np.sqrt(mean_squared_error(sim_values, fit_values))
#
#     # Fit linear regression for the trendline
#     reg = LinearRegression().fit(sim_values, fit_values)
#     trendline = reg.predict(sim_values)
#
#     # Store results
#     results.append({'Parameter': sim_param, 'Correlation': correlation, 'MAE': mae, 'RMSE': rmse})
#
#     # Plot recovery with color based on sigma
#     fig, ax = plt.subplots(1, 1, figsize=(6, 5))
#     scatter = ax.scatter(sim_values.flatten(), fit_values, c=sigma_values, cmap='viridis', alpha=0.8, label='Data')
#     plt.plot(sim_values, trendline, 'r--', label='Trendline')
#     plt.plot([sim_values.min(), sim_values.max()], [sim_values.min(), sim_values.max()], 'k-', label='Identity Line')
#     plt.colorbar(scatter, label='Sigma Value')
#     plt.xlabel('Simulated')
#     plt.ylabel('Recovered')
#     plt.title(f'Parameter Recovery: {sim_param} (r={correlation:.2f})')
#     plt.legend()
#     ax.spines[['top', 'right']].set_visible(False)
#     plt.show()
#
# # Visualize the results DataFrame
# print("Parameter Recovery Results:")
# print(pd.DataFrame(results))
# =============================================================================

