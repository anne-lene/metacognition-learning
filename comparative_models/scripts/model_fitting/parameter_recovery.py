# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:42:55 2025

@author: carll
"""

# Parameter Recovery

#%%
# parameter_recovery.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils import load_sim_df, load_fit_on_sim_df

# === Configuration ===
EXP = 2
OUTPUT_DIR = "recovery_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Model-specific simulated parameter names ===
model_param_map = {
    'bias': ['bias_mean', 'bias_sigma'],
    'wsls': ['wsls_sigma', 'wsls_win_boundary'],
    'rw': ['rw_alpha', 'rw_sigma'],
    'rw_cond': ['rw_cond_alpha_neut', 'rw_cond_alpha_pos', 'rw_cond_alpha_neg', 'rw_cond_sigma'],
    'ck': ['ck_alpha', 'ck_sigma', 'ck_beta'],
    'rwck': ['rwck_alpha', 'rwck_alpha_ck', 'rwck_sigma', 'rwck_sigma_ck', 'rwck_beta', 'rwck_beta_ck'],
    'rwp': ['rwp_alpha', 'rwp_sigma', 'rwp_w_rw', 'rwp_w_p', 'rwp_intercept'],
    'rwpd': ['rwpd_alpha', 'rwpd_sigma', 'rwpd_w_rw', 'rwpd_w_delta_p']
}

# === Mapping: sim param name -> fit param name ===
param_map = {
    'bias_mean': 'mean_bias',
    'bias_sigma': 'sd_bias',
    'wsls_sigma': 'std_WSLS',
    'wsls_win_boundary': 'win_boundary_WSLS',
    'rw_alpha': 'alpha_rw_symm',
    'rw_sigma': 'sigma_rw_symm',
    'rw_cond_alpha_neut': 'alpha_neut_rw_cond',
    'rw_cond_alpha_pos': 'alphaos_rw_cond',
    'rw_cond_alpha_neg': 'alpha_neg_rw_cond',
    'rw_cond_sigma': 'sigma_rw_cond',
    'ck_alpha': 'alpha_ck',
    'ck_sigma': 'sigma_ck',
    'ck_beta': 'beta_ck',
    'rwck_alpha': 'alpha_rwck',
    'rwck_alpha_ck': 'alpha_ck_rwck',
    'rwck_sigma': 'sigma_rwck',
    'rwck_sigma_ck': 'sigma_ck_rwck',
    'rwck_beta': 'beta_rwck',
    'rwck_beta_ck': 'ck_beta_rwck',
    'rwp_alpha': 'alpha_rwp',
    'rwp_sigma': 'sigma_rwp',
    'rwp_w_rw': 'w_rw_rwp',
    'rwp_w_p': 'w_performance_rwp',
    'rwp_intercept': 'intercept_rwp',
    'rwpd_alpha': 'alpha_rwpd',
    'rwpd_sigma': 'sigma_rwpd',
    'rwpd_w_rw': 'w_rw_rwpd',
    'rwpd_w_delta_p': 'w_pd_rwpd'
}

# === Load data ===
sim_data = load_sim_df(EXP=EXP)
fit_data = load_fit_on_sim_df(EXP=EXP)

# Standardize model names in fit data
model_name_map = {
    'win_stay': 'wsls',
    'delta_p_rw': 'rwpd'
}
fit_data['model'] = fit_data['model'].replace(model_name_map)

# === Build filtered sim data by model ===
summary_sim_rows = []
for model, param_list in model_param_map.items():
    if not param_list:
        continue
    model_rows = sim_data[sim_data.model == model].groupby(['pid', 'model'])[param_list].first().reset_index()
    summary_sim_rows.append(model_rows)

summary_sim_data = pd.concat(summary_sim_rows, ignore_index=True)

# === Merge with fit data ===
merged_data = pd.merge(summary_sim_data, fit_data, on=['pid', 'model'], suffixes=('_sim', '_fit'))

# === Recovery analysis function ===
def run_recovery_analysis(model_name, df, param_map, output_dir):
    results = []
    model_data = df[df.model == model_name]

    for sim_param, fit_param in param_map.items():
        if sim_param not in model_data.columns or fit_param not in model_data.columns:
            continue

        # Drop missing/inf values
        paired = model_data[[sim_param, fit_param]].replace([np.inf, -np.inf], np.nan).dropna()
        if paired.empty:
            print(f"⚠️ Skipping {sim_param} ({fit_param}) in model {model_name}: no valid data.")
            continue

        sim_vals = paired[sim_param].values.reshape(-1, 1)
        fit_vals = paired[fit_param].values

        # Compute metrics
        r, p = pearsonr(sim_vals.flatten(), fit_vals)
        mae = mean_absolute_error(sim_vals, fit_vals)
        rmse = np.sqrt(mean_squared_error(sim_vals, fit_vals))

        # Linear regression for trendline
        reg = LinearRegression().fit(sim_vals, fit_vals)
        trend = reg.predict(sim_vals)

        # Plot
        plt.figure(figsize=(5, 4))
        plt.scatter(sim_vals, fit_vals, alpha=0.6, label='Data')
        plt.plot(sim_vals, trend, 'r--', label='Trendline')
        plt.plot([sim_vals.min(), sim_vals.max()], [sim_vals.min(), sim_vals.max()], 'k-', label='Identity Line')
        plt.xlabel('Simulated')
        plt.ylabel('Recovered')
        plt.title(f'{sim_param} (r={r:.2f}, p={p:.3f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_{sim_param}_recovery.png")
        plt.close()

        results.append({
            'model': model_name,
            'sim_param': sim_param,
            'fit_param': fit_param,
            'correlation': r,
            'p_value': p,
            'mae': mae,
            'rmse': rmse
        })

    return pd.DataFrame(results)

# === Run recovery for each model ===
all_results = []
for model in model_param_map.keys():
    df = run_recovery_analysis(model, merged_data, param_map, OUTPUT_DIR)
    all_results.append(df)

# === Combine and save results ===
final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(f"{OUTPUT_DIR}/parameter_recovery_summary.csv", index=False)

print("✅ Parameter recovery complete!")
print(final_df)


#%%
# parameter_recovery.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.utils import load_sim_df, load_fit_on_sim_df
import os

# === Load data ===
sim_data = load_sim_df(EXP=2)
fit_data = load_fit_on_sim_df(EXP=2)

# === Standardize model names ===
model_name_map = {
    'win_stay': 'wsls',
    'delta_p_rw': 'rwpd'
}
fit_data['model'] = fit_data['model'].replace(model_name_map)

# Simulated parameter names per model
model_param_map = {
    'bias': ['bias_mean', 'bias_sigma'],
    'wsls': ['wsls_sigma', 'wsls_win_boundary'],
    'rw': ['rw_alpha', 'rw_sigma'],
    'rw_cond': ['rw_cond_alpha_neut', 'rw_cond_alpha_pos', 'rw_cond_alpha_neg', 'rw_cond_sigma'],
    'ck': ['ck_alpha', 'ck_sigma', 'ck_beta'],
    'rwck': ['rwck_alpha', 'rwck_alpha_ck', 'rwck_sigma', 'rwck_sigma_ck', 'rwck_beta', 'rwck_beta_ck'],
    'rwp': ['rwp_alpha', 'rwp_sigma', 'rwp_w_rw', 'rwp_w_p', 'rwp_intercept'],
    'rwpd': ['rwpd_alpha', 'rwpd_sigma', 'rwpd_w_rw', 'rwpd_w_delta_p']
}

# === Group sim data by pid and model, keep first value for each param ===
summary_sim_rows = []

for model, params in model_param_map.items():
    model_rows = sim_data[sim_data.model == model].groupby(['pid', 'model'])[params].first().reset_index()
    summary_sim_rows.append(model_rows)

summary_sim_data = pd.concat(summary_sim_rows, ignore_index=True)

# === Merge with fit data ===
merged_data = pd.merge(summary_sim_data, fit_data, on=['pid', 'model'], suffixes=('_sim', '_fit'))

# === Create output folder ===
output_dir = "recovery_results"
os.makedirs(output_dir, exist_ok=True)

# === Recovery analysis function ===
def run_recovery_analysis(model_name, df, param_map, output_dir):
    results = []
    model_data = df[df.model == model_name]

    for sim_param, fit_param in param_map.items():
        if sim_param not in model_data.columns or fit_param not in model_data.columns:
            continue  # Skip missing parameters

        # Combine into DataFrame and drop NaNs/Infs
        paired = model_data[[sim_param, fit_param]].replace([np.inf, -np.inf], np.nan).dropna()
        if paired.empty:
            continue  # skip this parameter if no valid data

        sim_vals = paired[sim_param].values.reshape(-1, 1)
        fit_vals = paired[fit_param].values

        # Metrics
        r, p = pearsonr(sim_vals.flatten(), fit_vals)
        mae = mean_absolute_error(sim_vals, fit_vals)
        rmse = np.sqrt(mean_squared_error(sim_vals, fit_vals))

        # Linear regression
        reg = LinearRegression().fit(sim_vals, fit_vals)
        trend = reg.predict(sim_vals)

        # Plot
        plt.figure(figsize=(5, 4))
        plt.scatter(sim_vals, fit_vals, alpha=0.6, label='Data')
        plt.plot(sim_vals, trend, 'r--', label='Trendline')
        plt.plot([sim_vals.min(), sim_vals.max()],
                 [sim_vals.min(), sim_vals.max()],
                 'k-', label='Identity Line')
        plt.xlabel('Simulated')
        plt.ylabel('Recovered')
        plt.title(f'{sim_param} (r={r:.2f}, p={p:.3f})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_{sim_param}_recovery.png")
        plt.close()

        results.append({
            'model': model_name,
            'parameter_sim': sim_param,
            'parameter_fit': fit_param,
            'correlation': r,
            'p_value': p,
            'mae': mae,
            'rmse': rmse
        })

    return pd.DataFrame(results)

# === Run for all relevant models in sim data ===
all_results = []
for model in merged_data['model'].unique():
    recovery_df = run_recovery_analysis(model, merged_data, param_map, output_dir)
    all_results.append(recovery_df)

# === Combine and save ===
final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv(f"{output_dir}/parameter_recovery_summary.csv", index=False)

print("✔️ Parameter recovery complete!")
print(final_results)


#%%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from src.utils import (load_sim_df, load_fit_on_sim_df)

# Load sim data
sim_data = load_sim_df(EXP=2)

# Load fit data
fit_data = load_fit_on_sim_df(EXP=2)

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
params = ['bias_mean', 'bias_sigma',
          'wsls_sigma', 'wsls_win_boundary',
          'rw_alpha', 'rw_sigma',
          'rw_cond_alpha_neut', 'rw_cond_alpha_pos', 'rw_cond_alpha_neg', 'rw_cond_sigma',
          'ck_alpha', 'ck_sigma', 'ck_beta',
          'rwck_alpha', 'rwck_alpha_ck', 'rwck_sigma', 'rwck_sigma_ck', 'rwck_beta', 'rwck_beta_ck',
          'rwp_alpha', 'rwp_sigma', 'rwp_w_rw', 'rwp_w_p', 'rwp_intercept',
          'rwpd_alpha', 'rwpd_sigma', 'rwpd_w_rw', 'rwpd_w_delta_p']

fit_data_params = ['mean_bias', 'sd_bias',
                   'std_WSLS','win_boundary_WSLS',
                   'alpha_rw_symm', 'sigma_rw_symm',
                   'alpha_neut_rw_cond', 'alphaos_rw_cond', 'alpha_neg_rw_cond', 'sigma_rw_cond',
                   'alpha_ck', 'sigma_ck', 'beta_ck',
                   'alpha_rwck', 'alpha_ck_rwck', 'sigma_rwck', 'sigma_ck_rwck','beta_rwck', 'ck_beta_rwck',
                   'alpha_rwp', 'sigma_rwp', 'w_rw_rwp', 'w_performance_rwp', 'intercept_rwp',
                   'alpha_rwpd', 'sigma_rwpd', 'w_rw_rwpd', 'w_pd_rwpd', 'intercept_rwpd']

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

