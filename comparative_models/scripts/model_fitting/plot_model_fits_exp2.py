# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:45:04 2024

@author: carll
"""


# Plotting Comparative Models for Experiment 2

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon, shapiro
from scipy.stats import sem
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from scipy.stats import stats
from src.utils import load_df

# Import data - Varied feedback condition (Experiment 2)
df=load_df(EXP=2)
# =============================================================================
# current_directory = os.path.dirname(os.path.abspath(__file__))
# parent_directory = os.path.dirname(current_directory)
# grandparent_directory = os.path.dirname(parent_directory)
# project_path = grandparent_directory
# experiment_data_path = r'variable_feedback/data'
# data_file = r'variable_fb_data_full_processed.csv'
# full_path = os.path.join(project_path, experiment_data_path, data_file)
# df = pd.read_csv(full_path, low_memory=False)
# =============================================================================


#%% Read the metrics from the Excel file and assign to variables

# =============================================================================
# local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
# working_dir = r'metacognition-learning\comparative_models'
# save_path = r'results\variable_feedback\model_comparison'
# name = 'model_metrics_CV_ORIGINAL.xlsx'
# save_path_full = os.path.join(local_folder, working_dir, save_path, name)
# =============================================================================

#df_m = pd.read_excel('EXP2_model_metrics_sessions_CV_v3_rand_column_change.xlsx')
df_m = pd.read_excel('C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/variable_feedback/model_comparison/models_fit_to_data/model_fits_EXP2_data.xlsx')
df_m = df_m.rename(columns=lambda col: col if col.endswith('_p') else f"{col}_p")

#%% Relative fit

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_random_p', 'aic_random_p', 'bic_random_p'],
    ['bias', 'nll_bias_p', 'aic_bias_p', 'bic_bias_p'],
    ['wsls', 'nll_win_stay_p', 'aic_win_stay_p', 'bic_win_stay_p'],
    #['rw_static', 'nll_rw_static_p', 'aic_rw_static_p', 'bic_rw_static_p'],
    ['rw', 'nll_rw_symm_p', 'aic_rw_symm_p', 'bic_rw_symm_p'],
    ['rw_cond', 'nll_rw_cond_p', 'aic_rw_cond_p', 'bic_rw_cond_p'],
    ['ck', 'nll_ck_p', 'aic_ck_p', 'bic_ck_p'],
    ['rwck', 'nll_rwck_p', 'aic_rwck_p', 'bic_rwck_p'],
    ['rwod', 'nll_rwpd_p', 'aic_rwpd_p', 'bic_rwpd_p'],
]

# Dictionary to store the results
results = {}

# Adjust the fits for each participant based on the best model for each metric
df_m['participant_id'] = df_m.index.values
participants = df_m['participant_id'].unique()
for participant in participants:
    for metric_index in range(1, 4):  # Index 1 for NLL, 2 for AIC, 3 for BIC
        min_metric_value = float('inf')
        best_model_metric_col = None

        # Find the best model based on the current metric for this participant
        for _, *metrics in models_metrics:
            participant_metric = df_m.loc[df_m['participant_id'] == participant, metrics[metric_index-1]].values
            if participant_metric.min() < min_metric_value:
                min_metric_value = participant_metric.min()
                best_model_metric_col = metrics[metric_index-1]

        # Subtract the best model's metric from all models' metrics for this participant and the current metric
        for _, *metrics in models_metrics:
            df_m.loc[df_m['participant_id'] == participant, metrics[metric_index-1]] -= min_metric_value

# Calculate mean and SEM for the adjusted metrics
for model, nll_col, aic_col, bic_col in models_metrics:
    # For NLL
    results[f'{model}_model_mean_nll'], results[f'{model}_model_sem_nll'] = calculate_mean_sem(df_m[nll_col])
    # For AIC
    results[f'{model}_model_mean_aic'], results[f'{model}_model_sem_aic'] = calculate_mean_sem(df_m[aic_col])
    # For BIC
    results[f'{model}_model_mean_bic'], results[f'{model}_model_sem_bic'] = calculate_mean_sem(df_m[bic_col])

    # Create variables for each array
    exec(f'{nll_col} = df_m["{nll_col}"].values')
    exec(f'{aic_col} = df_m["{aic_col}"].values')
    exec(f'{bic_col} = df_m["{bic_col}"].values')

    # Calculate mean and SEM
    mean_nll, sem_nll = calculate_mean_sem(eval(nll_col))
    mean_aic, sem_aic = calculate_mean_sem(eval(aic_col))
    mean_bic, sem_bic = calculate_mean_sem(eval(bic_col))

    # Create variables for mean and SEM
    exec(f'{model}_model_mean_nll = {mean_nll}')
    exec(f'{model}_model_sem_nll = {sem_nll}')
    exec(f'{model}_model_mean_aic = {mean_aic}')
    exec(f'{model}_model_sem_aic = {sem_aic}')
    exec(f'{model}_model_mean_bic = {mean_bic}')
    exec(f'{model}_model_sem_bic = {sem_bic}')

#%% Pairwise model comparison

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# Assuming nll_*_p are defined elsewhere
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "RW-Cond", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "RW-Cond": 'cyan',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

model_values = [
    nll_random_p, nll_bias_p, nll_win_stay_p,
    nll_rw_symm_p, nll_rw_cond_p, nll_ck_p,
    nll_rwck_p, nll_rwpd_p
]

# Perform pairwise statistical tests between all models
pairwise_results = []

for i in range(len(model_values)):
    for j in range(i + 1, len(model_values)):
        model1_name = model_names[i]
        model2_name = model_names[j]
        nll1 = model_values[i]
        nll2 = model_values[j]

        # Test for normality
        diff = nll1 - nll2
        _, p_value_normal = shapiro(diff)

        # Choose the test based on the normality of the differences
        if p_value_normal > 0.05:
            test_stat, p_value = ttest_rel(nll1, nll2)
            test_name = 'paired t-test'
        else:
            test_stat, p_value = wilcoxon(nll1, nll2)
            test_name = 'wilcoxon'
        print(f'{model1_name} vs {model2_name}, {test_name} p-value: {p_value}, t: {test_stat}')
        lower_model = model1_name if np.mean(nll1) < np.mean(nll2) else model2_name
        lower_model_color = color_mapping[lower_model]
        pairwise_results.append([model1_name, model2_name, test_name,
                                 test_stat, p_value,
                                 lower_model, lower_model_color])

# Convert pairwise results to DataFrame
pairwise_df = pd.DataFrame(pairwise_results, columns=['Model 1', 'Model 2',
                                                      'Test', 'Test Statistic',
                                                      'P-Value', 'Lower Model',
                                                      'Color'])

# Apply FDR correction
_, corrected_p_values, _, _ = multipletests(pairwise_df['P-Value'], alpha=0.05,
                                            method='fdr_bh')
pairwise_df['Corrected P-Value'] = corrected_p_values

# Create pivot tables for heatmap and color data
heatmap_data = pairwise_df.pivot(index='Model 1', columns='Model 2',
                                 values='P-Value')
color_data = pairwise_df.pivot(index='Model 1', columns='Model 2',
                               values='Color')
significance_data = pairwise_df.pivot(index='Model 1', columns='Model 2',
                                      values='Corrected P-Value')

# Complete the tables with all model names
heatmap_data = heatmap_data.reindex(index=model_names, columns=model_names)
color_data = color_data.reindex(index=model_names, columns=model_names)
significance_data = significance_data.reindex(index=model_names,
                                              columns=model_names)

# Fill diagonal with NaNs for visual clarity
np.fill_diagonal(heatmap_data.values, np.nan)
np.fill_diagonal(color_data.values, 'white')
np.fill_diagonal(significance_data.values, np.nan)

# Plot the heatmap with uncorrected p-values
plt.figure(figsize=(6, 6))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm',
                 linewidths=0.5, cbar=False,
                 mask=np.isnan(heatmap_data.values),
                 annot_kws={"color": "black", "size":10})

# Add color to the cells based on significance
for i in range(len(model_names)):
    for j in range(len(model_names)):
        if i < j:
            if (pd.notna(significance_data.iloc[i, j]) and
                significance_data.iloc[i, j] <= 0.05):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color='white',
                                           lw=0.5))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color=color_data.iloc[i, j],
                                           lw=0.5,
                                           alpha=0.5))
            else:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color='white',
                                           lw=0.5))

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[name], alpha=0.5)
           for name in model_names]
labels = model_names
plt.legend(handles, labels, title="Models", bbox_to_anchor=(1.05, 1.022),
           loc='upper left')

plt.xticks(rotation=45, ha='right')

plt.xlabel('')
plt.ylabel('')

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'Pairwise_model_comparison_relative_fit_CV_v10.svg'
save_path = os.path.join(result_path, file_name)

plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

#%% Stack model comparison means and hist in same figure.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, sem
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter, LogLocator
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import matplotlib.patches as patches

# Define the font size parameter
font_size = 16

# Update the default rc parameters for font size
plt.rcParams.update({'font.size': font_size})

# Prepare data for plotting
metric = [
    [nll_random_p,
     nll_bias_p,
     nll_win_stay_p,
     nll_rw_symm_p,
     nll_rw_cond_p,
     nll_ck_p,
     nll_rwck_p,
     nll_delta_p_rw_p]
]

model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "RW-Cond", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "RW-Cond": 'cyan',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

color_mapping_k = {name: "black" for name in model_names}

model_values = [
    nll_random_p, nll_bias_p, nll_win_stay_p,
    nll_rw_symm_p, nll_rw_cond_p,
    nll_ck_p, nll_rwck_p,
    nll_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])
df['Value'] = [i + 0.00001 for i in df['Value']]

# Prepare data for histogram
score_board = []
pids = []
for rand, bias, wsls, rw, rw_cond, ck, rwck, delta_p_rw, pid in zip(
                                                           metric[0][0],
                                                           metric[0][1],
                                                           metric[0][2],
                                                           metric[0][3],
                                                           metric[0][4],
                                                           metric[0][5],
                                                           metric[0][6],
                                                           metric[0][7],
                                                       range(len(metric[0][6]))
                                                           ):
    scores = np.array([rand, bias, wsls, rw, rw_cond, ck, rwck, delta_p_rw])
    min_score = np.min(scores)
    idxs = np.where(scores == min_score)[0]
    for idx in idxs:
        score_board.append(idx)
        pids.append(pid)

models = ['Random', 'Biased', 'Win-Stay-Loose-Shift', 'RW', 'RW-Cond',
          'Choice Kernel', 'RW + Choice Kernel', 'RW + Performance Delta']
counts = [score_board.count(0),  # Random
          score_board.count(1),  # Bias
          score_board.count(2),  # WSLS
          score_board.count(3),  # RW
          score_board.count(4),  # RW-Cond
          score_board.count(5),  # CK
          score_board.count(6),  # RWCK
          score_board.count(7)]  # RWPD

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                               height_ratios=[3,1])

# =============================================================================
# # First subplot: Violin plot
# subset_df = df[df['Metric'] == 'NLL']
# sns.violinplot(x="Model", y="Value", data=subset_df,
#                split=True, inner='quart', ax=ax1,
#                hue="Metric",
#                palette={"NLL": 'white'},
#                scale='width', width=0.5, linewidth=1,
#                bw_adjust=0.35,
#                dodge=False,
#                linecolor='lightgrey',
#                log_scale=True,
#                alpha=0.7)
#
# for i, model_name in enumerate(model_names):
#     model_data = subset_df[subset_df['Model'] == model_name]
#     x_positions = np.ones(len(model_data)) * i + 0.4
#     ax1.scatter(x_positions, model_data['Value'],
#                 color=color_mapping[model_name], s=4,
#                 label=model_name)
#     mean_value = model_data['Value'].mean()
#     sem_value = sem(model_data['Value'])
#     ax1.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
#                  color=color_mapping_k[model_name],
#                  capsize=5, markersize=4)
# =============================================================================

# First subplot: Violin plot with loop for each model
subset_df = df[df['Metric'] == 'NLL']

for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    sns.violinplot(x="Model", y="Value", data=model_data,
                   split=True, inner='quart', ax=ax1,
                   hue="Metric",
                   palette={"NLL": color_mapping[model_name]},
                   scale='width', width=0.5, linewidth=1,
                   bw_adjust=0.35,
                   dodge=False,
                   linecolor='darkgrey',
                   log_scale=True,
                   alpha=0.7)
    x_positions = np.ones(len(model_data)) * i + 0.4
    ax1.scatter(x_positions, model_data['Value'],
                color=color_mapping[model_name], s=4,
                label=model_name)
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])
    ax1.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                 color=color_mapping_k[model_name],
                 capsize=5, markersize=4)

ax1.set_xlabel('')
ax1.get_legend().remove()
ax1.set_ylabel('Cross-validated NLL')

# Adjust x-ticks by shifting them 0.25 to the right
current_xticks = ax1.get_xticks()
new_xticks = np.array(current_xticks) + 0.25
ax1.set_xticks(new_xticks)
ax1.set_xticklabels(model_names, rotation=45, ha='right')

def custom_log_formatter(x, pos):
    if x == 1e-5:
        return r'$0$'
    elif x == 1e-4:
        return r''
    elif x == 1e-3:
        return r'$10^{-3}$'
    else:
        return f'$10^{{{int(np.log10(x))}}}$'

ax1.set_yscale('log')
ax1.yaxis.set_major_formatter(FuncFormatter(custom_log_formatter))
ax1.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1,
                                          1e2]))
ax1.yaxis.set_minor_locator(LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9],
                                       numticks=12))
ax1.yaxis.set_minor_formatter(NullFormatter())

minor_ticks = [tick for tick in ax1.yaxis.get_minor_locator()()
               if tick > 1e-4]
minor_ticks.append(1e-4)
ax1.yaxis.set_minor_locator(FixedLocator(minor_ticks))

rect = plt.Rectangle((-0, 10**-4), len(model_names)+1, 0.0001,
                     facecolor="white", edgecolor="none", zorder=10)
ax1.add_patch(rect)

kwargs = dict(transform=ax1.transAxes, color='white', clip_on=False, lw=6)
ax1.plot((-0.015, 0.015), (0.25, 0.28), **kwargs)

rect = patches.Rectangle((0.08, 0.6), 0.1, 0.025, color='white', clip_on=False,
                         zorder=3)
trans = transforms.Affine2D().rotate_deg_around(0.5, 0.2575,
                                                45) + ax1.transAxes
rect.set_transform(trans)
ax1.add_patch(rect)


kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1,
              zorder=10)
ax1.plot((-0.0135, 0.015), (0.265, 0.295), **kwargs)
ax1.plot((-0.0135, 0.015), (0.225, 0.255), **kwargs)

# Load the corrected p-values from pairwise_df
pairwise_results = []
for model1 in model_names:
    for model2 in model_names:
        if model1 != model2:
            p_value = pairwise_df[(pairwise_df['Model 1'] == model1) & (pairwise_df['Model 2'] == model2)]['Corrected P-Value'].values
            if len(p_value) == 0:
                p_value = pairwise_df[(pairwise_df['Model 1'] == model2) & (pairwise_df['Model 2'] == model1)]['Corrected P-Value'].values
            if len(p_value) > 0:
                pairwise_results.append([model1, model2, p_value[0]])

pairwise_df_corrected = pd.DataFrame(pairwise_results, columns=['Model 1', 'Model 2', 'Corrected P-Value'])

def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value,
                          heightOffsetScalar=2.2, lengthScalar=0.2, text=True):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    """
    alpha = 0.05  # Significance threshold
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < alpha:
        if p_value < 0.001:
            sig_level = '***'
        elif p_value < 0.01:
            sig_level = '**'
        elif p_value < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    if text:
        ax.text((x1 + x2) / 2, y_max*0.80, sig_level, ha='center', va='bottom',
                fontsize=12, color='black')

# =============================================================================
# # Annotate significance bars for "RW + Performance Delta" model comparisons
# rw_perf_delta_index = model_names.index("RW + Performance Delta")
#
# for i, (model, y_offset) in enumerate(zip(model_names, np.logspace(2.6,
#                                                                    0.25,
#                                                                    8))):
#     if model == "Random":
#         y_offset = 100
#
#     if model != "RW + Performance Delta" :
#    # if model == "RW":
#         model_index = model_names.index(model)
#         corrected_p_value = pairwise_df_corrected[(pairwise_df_corrected['Model 1'] == "RW + Performance Delta") & (pairwise_df_corrected['Model 2'] == model)]['Corrected P-Value'].values
#         if len(corrected_p_value) == 0:
#             corrected_p_value = pairwise_df_corrected[(pairwise_df_corrected['Model 1'] == model) & (pairwise_df_corrected['Model 2'] == "RW + Performance Delta")]['Corrected P-Value'].values
#
#         if len(corrected_p_value) > 0 and corrected_p_value[0] <= 0.05:
#             y1_values = subset_df[subset_df['Model'] == "RW + Performance Delta"]['Value'].values
#             y2_values = subset_df[subset_df['Model'] == model]['Value'].values
#             annotate_significance(ax1,
#                                   rw_perf_delta_index + 0.4,
#                                   model_index + 0.4,
#                                   y1_values,
#                                   y2_values,
#                                   corrected_p_value[0],
#                                   heightOffsetScalar=y_offset,
#                                  # text=text_flag
#                                   )
# =============================================================================
# Define the base model for comparisons
base_model = "RW"  # Set this to the model name you want as the baseline

# Get the index of the base model in the list of model names
base_model_index = model_names.index(base_model)

# Annotate significance bars for comparisons with the base model
for i, (model, y_offset) in enumerate(zip(model_names, np.logspace(4.5, 0.25, 10))):
    if model == "Random":
        y_offset = 150
    if model == "Biased":
        y_offset = 800

    if model != base_model:
        model_index = model_names.index(model)

        # Look up the corrected p-value for the comparison with the base model
        corrected_p_value = pairwise_df_corrected[
            (pairwise_df_corrected['Model 1'] == base_model) &
            (pairwise_df_corrected['Model 2'] == model)
        ]['Corrected P-Value'].values

        # Check the reverse comparison if not found
        if len(corrected_p_value) == 0:
            corrected_p_value = pairwise_df_corrected[
                (pairwise_df_corrected['Model 1'] == model) &
                (pairwise_df_corrected['Model 2'] == base_model)
            ]['Corrected P-Value'].values

        # Annotate if the p-value is significant
        if len(corrected_p_value) > 0 and corrected_p_value[0] <= 0.05:
            y1_values = subset_df[subset_df['Model'] == base_model]['Value'].values
            y2_values = subset_df[subset_df['Model'] == model]['Value'].values
            annotate_significance(
                ax1,
                base_model_index + 0.4,
                model_index + 0.4,
                y1_values,
                y2_values,
                corrected_p_value[0],
                heightOffsetScalar=y_offset+5
            )
# =============================================================================
# y1_values = subset_df[subset_df['Model'] == "RW + Performance Delta"]['Value'].values
# y2_values = subset_df[subset_df['Model'] == model]['Value'].values
# y_max = max(max(y1_values), max(y2_values)) * 30.2
# ax1.text((rw_perf_delta_index + rw_perf_delta_index + 0.7) / 2, y_max*0.70,
#         '***', ha='center', va='bottom',
#         fontsize=19, color='black')
# =============================================================================

print('p_value_rwpd', round(corrected_p_value[0], 6))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: Histogram
bar_colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange',
              'silver']
bars = ax2.bar(np.arange(len(models)) + 0.25, counts, color=bar_colors)
ax2.set_xticks(np.arange(len(models)) + 0.25)  # Adjust x-ticks positions
ax2.set_ylabel('Best model count')
ax2.set_xlim(-1, len(models))  # Adjust xlim to ensure space for all bars
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout(pad=2.0)  # Increase padding to prevent overlap

# Adjust xlim
ax1.set_xlim(-0.5, 8)

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'combined_model_comparison_relative_fit_CV_v10.svg'
save_path = os.path.join(result_path, file_name)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

#%%

# Perform normality test for the differences
diff_winstay = nll_rw_symm_p - nll_delta_p_rw_p
diff_rw = nll_rw_symm_p - nll_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rw = shapiro(diff_rw)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_delta_p_rw_p,
                                   nll_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_delta_p_rw_p,
                                  nll_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rw > alpha:
    _, p_value_rwpd = ttest_rel(nll_delta_p_rw_p,
                                nll_rw_symm_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_delta_p_rw_p,
                               nll_rw_symm_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value,
                          num_comparisons, heightOffsetScalar=2.2,
                          lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    num_comparisons (int): Number of comparisons for Bonferroni correction.
    """
    alpha = 0.05 / num_comparisons  # Adjusted alpha for multiple comparisons
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < alpha:
        if p_value < 0.001:
            sig_level = '***'
        elif p_value < 0.01:
            sig_level = '**'
        elif p_value < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

annotate_significance(ax1, 2.4, 7.4,
                      nll_win_stay_p, nll_delta_p_rw_p,
                      p_value_winstay, num_comparisons, 20.5, 0.2)
annotate_significance(ax1, 3.4, 7.4,
                      nll_rw_symm_p, nll_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 5.6, 0.2)

print('p_value_rwpd', round(p_value_rwpd, 6))
print('p_value_winstay', round(p_value_winstay, 6))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: Histogram
bar_colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange',
              'silver']
bars = ax2.bar(np.arange(len(models)) + 0.25, counts, color=bar_colors)
ax2.set_xticks(np.arange(len(models)) + 0.25)  # Adjust x-ticks positions
ax2.set_ylabel('Best model count')
ax2.set_xlim(-1, len(models))  # Adjust xlim to ensure space for all bars
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout(pad=2.0)  # Increase padding to prevent overlap

# adjust xlim
ax1.set_xlim(-0.5, 8)

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'combined_model_comparison_relative_fit_CV.svg'
save_path = os.path.join(result_path, file_name)

#plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

#%% Plot the alpha of RWPD vs BDI

# Define the font size parameter
font_size = 14

# Update the default rc parameters for font size
plt.rcParams.update({'font.size': font_size})

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path,
                         data_file)
df = pd.read_csv(full_path, low_memory=False)

bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi_score.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.alpha_delta_p_rw_p[pid_rwpd_best.values]
    y = bdi[pid_rwpd_best.values]
else:
    x = df_m.alpha_delta_p_rw_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(x, y, label='Data points', color='#eeeeee',
           s=40, edgecolor='black', linewidths= 0.2, alpha=1)

# Add the regression line
ax.plot(x, slope * x + intercept, color='#666666', label='Regression line')


# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.35, 1.05),
            xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
           # bbox=dict(facecolor='white', edgecolor='black',
           #           alpha=0.5),
            )

ax.set_xlabel('RWPD Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(-0.025, 1)

# Add legend
#ax.legend()

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save the plot
#plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% Plot the alpha of RW vs BDI
import scipy.stats as stats

# Define the font size parameter
font_size = 14

# Update the default rc parameters for font size
plt.rcParams.update({'font.size': font_size})

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi_score.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.alpha_rw_symm_p[pid_rw_best.values]
    y = bdi[pid_rw_best.values]
else:
    x = df_m.alpha_rw_symm_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(x, y, label='Data points', color='#CBC3E3',
           s=40, edgecolor='black', linewidths= 0.2,)

# Add the regression line
ax.plot(x, slope * x + intercept, color='purple', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.35, 1.05), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
           # bbox=dict(facecolor='white', edgecolor='black', alpha=0.5),
           )

ax.set_xlabel('RW Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(-0.025, 1)
# Add legend
#ax.legend()

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'RW_alpha_vs_BDI_v10.svg'
save_path = os.path.join(project_path,
                         'comparative_models',
                         result_path,
                         file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% metacog_bias vs alpha
import statsmodels.api as sm

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

# Step 1: Calculate metacognitive bias (intercept) for each participant
df['abs_error'] = abs(df['feedback'] - df['estimate'])
df['prev_feedback'] = df.groupby('pid')['feedback'].shift(1)
df = df.dropna(subset=['abs_error', 'prev_feedback', 'confidence'])

metacog_bias = {}
for pid, participant_data in df.groupby('pid'):
    X = participant_data[['abs_error', 'prev_feedback']]
    X = sm.add_constant(X)  # Add intercept
    y = participant_data['confidence']
    model = sm.OLS(y, X).fit()
    metacog_bias[pid] = model.params['const']  # Intercept as metacog bias

# Convert metacog_bias to DataFrame
metacog_bias_df = pd.DataFrame(list(metacog_bias.items()), columns=['pid', 'metacog_bias'])

# Step 2: Extract RW learning rate (alpha) and BDI scores
bdi = []
alpha = []
for participant in tqdm(df['pid'].unique()):
    bdi_score = df[df['pid'] == participant].bdi_score.unique()[0]
    alpha_value = df_m[df_m['pid_p'] == participant].alpha_rw_symm_p.values[0]  # Assuming alpha is stored here
    bdi.append(bdi_score)
    alpha.append(alpha_value)

# Convert to DataFrame
rw_df = pd.DataFrame({'pid': df['pid'].unique(), 'alpha': alpha, 'bdi_score': bdi})

# Step 3: Merge metacognitive bias with RW learning rate and BDI data
merged_df = pd.merge(rw_df, metacog_bias_df, on='pid')

# Step 4: Perform linear regression between RW learning rate (alpha) and metacognitive bias
x = merged_df['alpha']
y = merged_df['metacog_bias']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Step 5: Plot RW learning rate vs. Metacognitive Bias with regression line
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.scatter(x, y, label='Data points', color='#CBC3E3', s=40, edgecolor='black', linewidths=0.2)
ax.plot(x, slope * x + intercept, color='purple', label='Regression line')

# Annotate with R^2 and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.35, 1.05), xycoords='axes fraction', fontsize=12, ha='left', va='top')

# Set plot labels and limits
ax.set_xlabel('RW Learning rate')
ax.set_ylabel('Metacognitive Bias')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(-0.025, 1)

# Save path for the plot
result_path = r"results\variable_feedback\model_comparison"
file_name = 'RW_alpha_vs_Metacognitive_Bias_v10.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Optional: Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%%

local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
working_dir = r'metacognition-learning\comparative_models'
save_path = r'results\variable_feedback\model_comparison'
name = 'model_metrics_CV_ORIGINAL.xlsx'
save_path_full = os.path.join(local_folder, working_dir, save_path, name)
df_m = pd.read_excel('EXP2_model_metrics_sessions_CV_v10.xlsx')

#%%# Function to calculate mean and SEM of the absolute model fit

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# Dictionary to store the results
results = {}

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_random_p', 'aic_random_p', 'bic_random_p'],
    ['bias', 'nll_bias_p', 'aic_bias_p', 'bic_bias_p'],
    ['win_stay', 'nll_win_stay_p', 'aic_win_stay_p', 'bic_win_stay_p'],
    #['rw_static', 'nll_rw_static_p', 'aic_rw_static_p', 'bic_rw_static_p'],
    ['rw_symm', 'nll_rw_symm_p', 'aic_rw_symm_p', 'bic_rw_symm_p'],
    ['rw_cond', 'nll_rw_cond_p', 'aic_rw_cond_p', 'bic_rw_cond_p'],
    ['ck', 'nll_ck_p', 'aic_ck_p', 'bic_ck_p'],
    ['rwck', 'nll_rwck_p', 'aic_rwck_p', 'bic_rwck_p'],
    ['delta_p_rw', 'nll_delta_p_rw_p', 'aic_delta_p_rw_p', 'bic_delta_p_rw_p'],
]

# Loop through each model and metric, calculate mean and SEM, and store in results dictionary
for model, nll_col, aic_col, bic_col in models_metrics:
    results[f'{model}_model_mean_nll'], results[f'{model}_model_sem_nll'] = calculate_mean_sem(df_m[nll_col])
    results[f'{model}_model_mean_aic'], results[f'{model}_model_sem_aic'] = calculate_mean_sem(df_m[aic_col])
    results[f'{model}_model_mean_bic'], results[f'{model}_model_sem_bic'] = calculate_mean_sem(df_m[bic_col])

# Loop through each model and metric, calculate mean and SEM, and create named variables
for model, nll_col, aic_col, bic_col in models_metrics:
    # Create variables for each array
    exec(f'{nll_col} = df_m["{nll_col}"].values')
    exec(f'{aic_col} = df_m["{aic_col}"].values')
    exec(f'{bic_col} = df_m["{bic_col}"].values')

    # Calculate mean and SEM
    mean_nll, sem_nll = calculate_mean_sem(eval(nll_col))
    mean_aic, sem_aic = calculate_mean_sem(eval(aic_col))
    mean_bic, sem_bic = calculate_mean_sem(eval(bic_col))

    # Create variables for mean and SEM
    exec(f'{model}_model_mean_nll = {mean_nll}')
    exec(f'{model}_model_sem_nll = {sem_nll}')
    exec(f'{model}_model_mean_aic = {mean_aic}')
    exec(f'{model}_model_sem_aic = {sem_aic}')
    exec(f'{model}_model_mean_bic = {mean_bic}')
    exec(f'{model}_model_sem_bic = {sem_bic}')

#%% Pairwise model comparison

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# Assuming nll_*_p are defined elsewhere
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "RW-Cond", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "RW-Cond": 'cyan',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

model_values = [
    nll_random_p, nll_bias_p, nll_win_stay_p,
    nll_rw_symm_p, nll_rw_cond_p, nll_ck_p,
    nll_rwck_p, nll_delta_p_rw_p
]

# Perform pairwise statistical tests between all models
pairwise_results = []

for i in range(len(model_values)):
    for j in range(i + 1, len(model_values)):
        model1_name = model_names[i]
        model2_name = model_names[j]
        nll1 = model_values[i]
        nll2 = model_values[j]

        # Test for normality
        diff = nll1 - nll2
        _, p_value_normal = shapiro(diff)

        # Choose the test based on the normality of the differences
        if p_value_normal > 0.05:
            test_stat, p_value = ttest_rel(nll1, nll2)
            test_name = 'paired t-test'
        else:
            test_stat, p_value = wilcoxon(nll1, nll2)
            test_name = 'wilcoxon'

        lower_model = model1_name if np.mean(nll1) < np.mean(nll2) else model2_name
        lower_model_color = color_mapping[lower_model]
        pairwise_results.append([model1_name, model2_name, test_name,
                                 test_stat, p_value,
                                 lower_model, lower_model_color])

# Convert pairwise results to DataFrame
pairwise_df = pd.DataFrame(pairwise_results, columns=['Model 1', 'Model 2',
                                                      'Test', 'Test Statistic',
                                                      'P-Value', 'Lower Model',
                                                      'Color'])

# Apply FDR correction
_, corrected_p_values, _, _ = multipletests(pairwise_df['P-Value'], alpha=0.05,
                                            method='fdr_bh')
pairwise_df['Corrected P-Value'] = corrected_p_values

# Create pivot tables for heatmap and color data
heatmap_data = pairwise_df.pivot(index='Model 1', columns='Model 2',
                                 values='P-Value')
color_data = pairwise_df.pivot(index='Model 1', columns='Model 2',
                               values='Color')
significance_data = pairwise_df.pivot(index='Model 1', columns='Model 2',
                                      values='Corrected P-Value')

# Complete the tables with all model names
heatmap_data = heatmap_data.reindex(index=model_names, columns=model_names)
color_data = color_data.reindex(index=model_names, columns=model_names)
significance_data = significance_data.reindex(index=model_names,
                                              columns=model_names)

# Fill diagonal with NaNs for visual clarity
np.fill_diagonal(heatmap_data.values, np.nan)
np.fill_diagonal(color_data.values, 'white')
np.fill_diagonal(significance_data.values, np.nan)

# Plot the heatmap with uncorrected p-values
plt.figure(figsize=(6, 6))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm',
                 linewidths=0.5, cbar=False,
                 mask=np.isnan(heatmap_data.values),
                 annot_kws={"color": "black", "size":10})

# Add color to the cells based on significance
for i in range(len(model_names)):
    for j in range(len(model_names)):
        if i < j:
            if (pd.notna(significance_data.iloc[i, j]) and
                significance_data.iloc[i, j] <= 0.05):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color='white',
                                           lw=0.5))
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color=color_data.iloc[i, j],
                                           lw=0.5,
                                           alpha=0.5))
            else:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                           color='white',
                                           lw=0.5))

# Add legend
handles = [plt.Rectangle((0, 0), 1, 1, color=color_mapping[name], alpha=0.5)
           for name in model_names]
labels = model_names
plt.legend(handles, labels, title="Models", bbox_to_anchor=(1.05, 1.022),
           loc='upper left')

plt.xticks(rotation=45, ha='right')

plt.xlabel('')
plt.ylabel('')

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'Pairwise_model_comparison_relative_fit_CV_v10.svg'
save_path = os.path.join(result_path, file_name)

plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

#%% Stack model comparison means and hist in same figure.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, sem
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter, LogLocator
from matplotlib.patches import Rectangle
import matplotlib.transforms as transforms
import matplotlib.patches as patches

# Define the font size parameter
font_size = 16

# Update the default rc parameters for font size
plt.rcParams.update({'font.size': font_size})

# Prepare data for plotting
metric = [
    [nll_random_p,
     nll_bias_p,
     nll_win_stay_p,
     nll_rw_symm_p,
     nll_rw_cond_p,
     nll_ck_p,
     nll_rwck_p,
     nll_delta_p_rw_p]
]

model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "RW-Cond", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "RW-Cond": 'cyan',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

color_mapping_k = {name: "black" for name in model_names}

model_values = [
    nll_random_p, nll_bias_p, nll_win_stay_p,
    nll_rw_symm_p, nll_rw_cond_p,
    nll_ck_p, nll_rwck_p,
    nll_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])
df['Value'] = [i + 0.00001 for i in df['Value']]

# Prepare data for histogram
score_board = []
pids = []
for rand, bias, wsls, rw, rw_cond, ck, rwck, delta_p_rw, pid in zip(
                                                           metric[0][0],
                                                           metric[0][1],
                                                           metric[0][2],
                                                           metric[0][3],
                                                           metric[0][4],
                                                           metric[0][5],
                                                           metric[0][6],
                                                           metric[0][7],
                                                       range(len(metric[0][6]))
                                                           ):
    scores = np.array([rand, bias, wsls, rw, rw_cond, ck, rwck, delta_p_rw])
    min_score = np.min(scores)
    idxs = np.where(scores == min_score)[0]
    for idx in idxs:
        score_board.append(idx)
        pids.append(pid)

models = ['Random', 'Biased', 'Win-Stay-Loose-Shift', 'RW', 'RW-Cond',
          'Choice Kernel', 'RW + Choice Kernel', 'RW + Performance Delta']
counts = [score_board.count(0),  # Random
          score_board.count(1),  # Bias
          score_board.count(2),  # WSLS
          score_board.count(3),  # RW
          score_board.count(4),  # RW-Cond
          score_board.count(5),  # CK
          score_board.count(6),  # RWCK
          score_board.count(7)]  # RWPD

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                               height_ratios=[3,1])

# =============================================================================
# # First subplot: Violin plot
# subset_df = df[df['Metric'] == 'NLL']
# sns.violinplot(x="Model", y="Value", data=subset_df,
#                split=True, inner='quart', ax=ax1,
#                hue="Metric",
#                palette={"NLL": 'white'},
#                scale='width', width=0.5, linewidth=1,
#                bw_adjust=0.35,
#                dodge=False,
#                linecolor='lightgrey',
#                log_scale=True,
#                alpha=0.7)
#
# for i, model_name in enumerate(model_names):
#     model_data = subset_df[subset_df['Model'] == model_name]
#     x_positions = np.ones(len(model_data)) * i + 0.4
#     ax1.scatter(x_positions, model_data['Value'],
#                 color=color_mapping[model_name], s=4,
#                 label=model_name)
#     mean_value = model_data['Value'].mean()
#     sem_value = sem(model_data['Value'])
#     ax1.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
#                  color=color_mapping_k[model_name],
#                  capsize=5, markersize=4)
# =============================================================================

# First subplot: Violin plot with loop for each model
subset_df = df[df['Metric'] == 'NLL']

for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    sns.violinplot(x="Model", y="Value", data=model_data,
                   split=True, inner='quart', ax=ax1,
                   hue="Metric",
                   palette={"NLL": color_mapping[model_name]},
                   scale='width', width=0.5, linewidth=1,
                   bw_adjust=0.35,
                   dodge=False,
                   linecolor='darkgrey',
                   log_scale=True,
                   alpha=0.7)
    x_positions = np.ones(len(model_data)) * i + 0.4
    ax1.scatter(x_positions, model_data['Value'],
                color=color_mapping[model_name], s=4,
                label=model_name)
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])
    ax1.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                 color=color_mapping_k[model_name],
                 capsize=5, markersize=4)

ax1.set_xlabel('')
ax1.get_legend().remove()
ax1.set_ylabel('Cross-validated NLL')

# Adjust x-ticks by shifting them 0.25 to the right
current_xticks = ax1.get_xticks()
new_xticks = np.array(current_xticks) + 0.25
ax1.set_xticks(new_xticks)
ax1.set_xticklabels(model_names, rotation=45, ha='right')

def custom_log_formatter(x, pos):
    if x == 1e-5:
        return r'$0$'
    elif x == 1e-4:
        return r''
    elif x == 1e-3:
        return r'$10^{-3}$'
    else:
        return f'$10^{{{int(np.log10(x))}}}$'

# =============================================================================
# ax1.set_yscale('log')
# ax1.yaxis.set_major_formatter(FuncFormatter(custom_log_formatter))
# ax1.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1,
#                                           1e2]))
# ax1.yaxis.set_minor_locator(LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9],
#                                        numticks=12))
# ax1.yaxis.set_minor_formatter(NullFormatter())
#
# minor_ticks = [tick for tick in ax1.yaxis.get_minor_locator()()
#                if tick > 1e-4]
# minor_ticks.append(1e-4)
# ax1.yaxis.set_minor_locator(FixedLocator(minor_ticks))
#
# rect = plt.Rectangle((-0, 10**-4), len(model_names)+1, 0.0001,
#                      facecolor="white", edgecolor="none", zorder=10)
# ax1.add_patch(rect)
#
# kwargs = dict(transform=ax1.transAxes, color='white', clip_on=False, lw=6)
# ax1.plot((-0.015, 0.015), (0.25, 0.28), **kwargs)
#
# rect = patches.Rectangle((0.08, 0.6), 0.1, 0.025, color='white', clip_on=False,
#                          zorder=3)
# trans = transforms.Affine2D().rotate_deg_around(0.5, 0.2575,
#                                                 45) + ax1.transAxes
# rect.set_transform(trans)
# ax1.add_patch(rect)
#
#
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1,
#               zorder=10)
# ax1.plot((-0.0135, 0.015), (0.265, 0.295), **kwargs)
# ax1.plot((-0.0135, 0.015), (0.225, 0.255), **kwargs)
#
# =============================================================================
# Load the corrected p-values from pairwise_df
pairwise_results = []
for model1 in model_names:
    for model2 in model_names:
        if model1 != model2:
            p_value = pairwise_df[(pairwise_df['Model 1'] == model1) & (pairwise_df['Model 2'] == model2)]['Corrected P-Value'].values
            if len(p_value) == 0:
                p_value = pairwise_df[(pairwise_df['Model 1'] == model2) & (pairwise_df['Model 2'] == model1)]['Corrected P-Value'].values
            if len(p_value) > 0:
                pairwise_results.append([model1, model2, p_value[0]])

pairwise_df_corrected = pd.DataFrame(pairwise_results, columns=['Model 1', 'Model 2', 'Corrected P-Value'])

def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value,
                          heightOffsetScalar=2.2, lengthScalar=0.2, text=True):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    """
    alpha = 0.05  # Significance threshold
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < alpha:
        if p_value < 0.001:
            sig_level = '***'
        elif p_value < 0.01:
            sig_level = '**'
        elif p_value < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    if text:
        ax.text((x1 + x2) / 2, y_max*0.70, sig_level, ha='center', va='bottom',
                fontsize=12, color='black')

# Annotate significance bars for "RW + Performance Delta" model comparisons
rw_perf_delta_index = model_names.index("RW + Performance Delta")

for i, (model, y_offset) in enumerate(zip(model_names, np.logspace(2.6,
                                                                   0.25,
                                                                   8))):
    if model == "Random":
        y_offset = 100

    if model != "RW + Performance Delta" :
   # if model == "RW":
        model_index = model_names.index(model)
        corrected_p_value = pairwise_df_corrected[(pairwise_df_corrected['Model 1'] == "RW + Performance Delta") & (pairwise_df_corrected['Model 2'] == model)]['Corrected P-Value'].values
        if len(corrected_p_value) == 0:
            corrected_p_value = pairwise_df_corrected[(pairwise_df_corrected['Model 1'] == model) & (pairwise_df_corrected['Model 2'] == "RW + Performance Delta")]['Corrected P-Value'].values

        if len(corrected_p_value) > 0 and corrected_p_value[0] <= 0.05:
            y1_values = subset_df[subset_df['Model'] == "RW + Performance Delta"]['Value'].values
            y2_values = subset_df[subset_df['Model'] == model]['Value'].values
            annotate_significance(ax1,
                                  rw_perf_delta_index + 0.4,
                                  model_index + 0.4,
                                  y1_values,
                                  y2_values,
                                  corrected_p_value[0],
                                  heightOffsetScalar=y_offset,
                                 # text=text_flag
                                  )

# =============================================================================
# y1_values = subset_df[subset_df['Model'] == "RW + Performance Delta"]['Value'].values
# y2_values = subset_df[subset_df['Model'] == model]['Value'].values
# y_max = max(max(y1_values), max(y2_values)) * 30.2
# ax1.text((rw_perf_delta_index + rw_perf_delta_index + 0.7) / 2, y_max*0.70,
#         '***', ha='center', va='bottom',
#         fontsize=19, color='black')
# =============================================================================

print('p_value_rwpd', round(corrected_p_value[0], 6))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: Histogram
bar_colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange',
              'silver']
bars = ax2.bar(np.arange(len(models)) + 0.25, counts, color=bar_colors)
ax2.set_xticks(np.arange(len(models)) + 0.25)  # Adjust x-ticks positions
ax2.set_ylabel('Best model count')
ax2.set_xlim(-1, len(models))  # Adjust xlim to ensure space for all bars
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout(pad=2.0)  # Increase padding to prevent overlap

# Adjust xlim
ax1.set_xlim(-0.5, 8)

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'combined_model_comparison_relative_fit_CV_v10.svg'
save_path = os.path.join(result_path, file_name)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

#%%

# Perform normality test for the differences
diff_winstay = nll_rw_symm_p - nll_delta_p_rw_p
diff_rw = nll_rw_symm_p - nll_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rw = shapiro(diff_rw)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_delta_p_rw_p,
                                   nll_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_delta_p_rw_p,
                                  nll_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rw > alpha:
    _, p_value_rwpd = ttest_rel(nll_delta_p_rw_p,
                                nll_rw_symm_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_delta_p_rw_p,
                               nll_rw_symm_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value,
                          num_comparisons, heightOffsetScalar=2.2,
                          lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    num_comparisons (int): Number of comparisons for Bonferroni correction.
    """
    alpha = 0.05 / num_comparisons  # Adjusted alpha for multiple comparisons
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < alpha:
        if p_value < 0.001:
            sig_level = '***'
        elif p_value < 0.01:
            sig_level = '**'
        elif p_value < 0.05:
            sig_level = '*'
        else:
            sig_level = 'ns'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

annotate_significance(ax1, 2.4, 7.4,
                      nll_win_stay_p, nll_delta_p_rw_p,
                      p_value_winstay, num_comparisons, 20.5, 0.2)
annotate_significance(ax1, 3.4, 7.4,
                      nll_rw_symm_p, nll_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 5.6, 0.2)

print('p_value_rwpd', round(p_value_rwpd, 6))
print('p_value_winstay', round(p_value_winstay, 6))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: Histogram
bar_colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange',
              'silver']
bars = ax2.bar(np.arange(len(models)) + 0.25, counts, color=bar_colors)
ax2.set_xticks(np.arange(len(models)) + 0.25)  # Adjust x-ticks positions
ax2.set_ylabel('Best model count')
ax2.set_xlim(-1, len(models))  # Adjust xlim to ensure space for all bars
ax2.set_xticklabels(models, rotation=45, ha='right')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

plt.tight_layout(pad=2.0)  # Increase padding to prevent overlap

# adjust xlim
ax1.set_xlim(-0.5, 8)

# Set save path
result_path = r"results\variable_feedback\model_comparison"
file_name = 'combined_model_comparison_relative_fit_CV.svg'
save_path = os.path.join(result_path, file_name)

plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.show()

