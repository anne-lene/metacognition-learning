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
from src.utility_functions import add_session_column
from scipy.stats import stats
from src.models import (fit_model,
                        fit_model_with_cv,
                        fit_random_model,
                        random_model,
                        random_model_w_bias,
                        win_stay_lose_shift,
                        rw_symmetric_LR,
                        choice_kernel,
                        RW_choice_kernel,
                        delta_P_RW)

# Import data - Varied feedback condition (Experiment 2)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)


#%% Read the metrics from the Excel file and assign to variables

local_folder = r'C:\Users\carll\OneDrive\Skrivbord\Oxford\DPhil'
working_dir = r'metacognition-learning\comparative_models'
save_path = r'results\variable_feedback\model_comparison'
name = 'model_metrics_CV_ORIGINAL.xlsx'
save_path_full = os.path.join(local_folder, working_dir, save_path, name)
df_m = pd.read_excel(save_path_full)

#%% Relative fit

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_array_random_p', 'aic_array_random_p', 'bic_array_random_p'],
    ['bias', 'nll_array_bias_p', 'aic_array_bias_p', 'bic_array_bias_p'],
    ['win_stay', 'nll_array_win_stay_p', 'aic_array_win_stay_p', 'bic_array_win_stay_p'],
    #['rw_static', 'nll_array_rw_static_p', 'aic_array_rw_static_p', 'bic_array_rw_static_p'],
    ['rw_symm', 'nll_array_rw_symm_p', 'aic_array_rw_symm_p', 'bic_array_rw_symm_p'],
    ['rw_cond', 'nll_array_rw_cond_p', 'aic_array_rw_cond_p', 'bic_array_rw_cond_p'],
    ['ck', 'nll_array_ck_p', 'aic_array_ck_p', 'bic_array_ck_p'],
    ['rwck', 'nll_array_rwck_p', 'aic_array_rwck_p', 'bic_array_rwck_p'],
    ['delta_p_rw', 'nll_array_delta_p_rw_p', 'aic_array_delta_p_rw_p', 'bic_array_delta_p_rw_p'],
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

#%% Only NLL - Boxplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import shapiro, ttest_rel, wilcoxon
import os

def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value,
                          num_comparisons, heightOffsetScalar=2.2,
                          lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot with Bonferroni correction.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    num_comparisons (int): The number of comparisons being performed for Bonferroni correction.
    """


    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    # Adjust alpha levels for Bonferroni correction
    if p_value < 0.001 / num_comparisons:
        sig_level = '***'
    elif p_value < 0.01 / num_comparisons:
        sig_level = '**'
    elif p_value < 0.05 / num_comparisons:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

def darken_color(color, amount=0.5):
    """
    Darken a given color.
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([max(0, min(1, c[i] * (1 - amount))) for i in range(3)])

fig, ax = plt.subplots(figsize=(5, 5))  # Increase the figure size for better readability

model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "RW-Cond","Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]
# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange', 'silver']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # Markers for each model

# Define x-coordinates and offsets for each model within the group
x = np.arange(0, 1)  # Base x-coordinates for metrics
offset = 0.35  # Offset for each model within a group

# Plotting NLL for each model
model_values = [
    nll_array_random_p, nll_array_bias_p,
    nll_array_win_stay_p, nll_array_rw_symm_p,
    nll_array_rw_cond_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[0] + offset * (i - 3)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black'), zorder=0,
        )

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p


_, p_value_normal_winstay = shapiro(diff_winstay)

_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2 # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (-1),
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 1.4, 0.1)

annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (3),
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 2.25, 0.035)

# Customizing the Axes
ax.set_xticks([x[0] + offset * (i - 3) for i in range(len(model_names))])
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('Cross-validated NLL')
#ax.set_yscale('log')  # Set y-axis to log scale
ax.set_xlim(-1.3, 1.3)
# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

import matplotlib.lines as mlines

# Custom legend handles
custom_handles = [
    mlines.Line2D([], [], color=color, marker='o',
                  linestyle='None', markersize=6,
                  markerfacecolor=color,
                  alpha=1)
    for color in colors
]

# =============================================================================
# # Adding a legend
# ax.legend(custom_handles, model_names, scatterpoints=1, markerscale=1.2,
#           fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
# =============================================================================

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()

#%% KDE + Boxplot + Scatter + mean+-sem

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "RW-Cond", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "RW-Cond":'cyan',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}
color_mapping_k = {name: "black" for name in model_names}

# Assuming nll_array_*_p are defined elsewhere
model_values = [
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_rw_cond_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Plot NLL metric
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

subset_df = df[df['Metric'] == 'NLL']

# Debugging print statements
print("subset_df head:\n", subset_df.head())
print("subset_df describe:\n", subset_df.describe())
print("subset_df info:\n", subset_df.info())

# Violin plot for the left side
sns.violinplot(x="Model", y="Value", data=subset_df,
               split=True, inner='quart', ax=ax,
               hue="Metric",
               palette={"NLL": 'lightblue'},
               scale='width', width=0.5, linewidth=1,
               bw_adjust=0.35,
               dodge=False,
              # inner_kws = {'whis_width':1,
              #              'box_width': 5,
              #              'marker': '_'}
             )

# =============================================================================
# # Add Boxplot with slight offset to the right
# positions = range(len(model_names))
# offset = 0.25
# positions = [pos + offset for pos in positions]
# sns.boxplot(x="Model", y="Value", data=df, width=0.1, positions=positions,
#             color='lightgrey', fliersize=0, linewidth=1,
#             medianprops=dict(marker='', color='black', markerfacecolor='white',
#                              markersize=5))
# =============================================================================

# Scatter plot for data points with error bars
for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    x_positions = np.ones(len(model_data)) * i + 0.4
    ax.scatter(x_positions, model_data['Value'],
               color=color_mapping[model_name], s=1,
               label=model_name)

    # Calculate mean and SEM
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])

    # Plot error bars
    ax.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                color=color_mapping_k[model_name],
                capsize=4, markersize=3)

ax.set_xlabel('')
ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
ax.set_ylabel('Cross-validated NLL')


# Adjust x-tick positions and rotation of labels
current_xticks = ax.get_xticks()
new_xticks = np.array(current_xticks) + 0.4
ax.set_xticks(new_xticks)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#ax.set_yscale('log')  # Set y-axis to log scale

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
annotate_significance(ax, 3.4, 2.4,
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 5.2, 0.05)
annotate_significance(ax, 3.4, 6.4,
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 2.3, 0.035)

plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1_violin.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()


#%% KDE + Boxplot + Scatter + mean+-sem

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, sem

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

color_mapping_k = {name: "black" for name in model_names}

# Assuming nll_array_*_p are defined elsewhere
model_values = [
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Remove NaNs introduced by log transformation handling
#df.dropna(subset=["Value"], inplace=True)

# Apply log transformation with a small shift
#df['Value'] = np.log10(df['Value'] + 1e-10)
#df['Value'] = abs(df['Value'])
df['Value'] = [i + 0.0001 for i in df['Value']]

# Plot NLL metric
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

subset_df = df[df['Metric'] == 'NLL']

# Debugging print statements
print("subset_df head:\n", subset_df.head())
print("subset_df describe:\n", subset_df.describe())
print("subset_df info:\n", subset_df.info())

# Violin plot for the left side
sns.violinplot(x="Model", y="Value", data=subset_df,
               split=True, inner='quart', ax=ax,
               hue="Metric",
               palette={"NLL": 'lightblue'},
               scale='width', width=0.5, linewidth=1,
               bw_adjust=0.35,
               dodge=False,
               log_scale=True
               )

# Scatter plot for data points with error bars
for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    x_positions = np.ones(len(model_data)) * i + 0.4
    ax.scatter(x_positions, model_data['Value'],
               color=color_mapping[model_name], s=1,
               label=model_name)

    # Calculate mean and SEM
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])

    # Plot error bars
    ax.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                color=color_mapping_k[model_name],
                capsize=4, markersize=3)

ax.set_xlabel('')
ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
ax.set_ylabel('Cross-validated NLL')

# Adjust x-tick positions and rotation of labels
current_xticks = ax.get_xticks()
new_xticks = np.array(current_xticks) + 0.4
ax.set_xticks(new_xticks)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yscale('log')  # Set y-axis to log scale

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value, num_comparisons, heightOffsetScalar=2.2, lengthScalar=0.05):
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

annotate_significance(ax, 3.4, 2.4,
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 3.2, 0.2)
annotate_significance(ax, 3.4, 6.4,
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 3.5, 0.2)

plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1_violin_log.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.show()

#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon, sem
from matplotlib.ticker import FuncFormatter, FixedLocator, NullFormatter, LogLocator
from matplotlib.patches import Rectangle

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

color_mapping_k = {name: "black" for name in model_names}

# Assuming nll_array_*_p are defined elsewhere
model_values = [
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Remove NaNs and add a small shift to avoid log(0)
df['Value'] = [i + 0.00001 for i in df['Value']]

# Plot NLL metric
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

subset_df = df[df['Metric'] == 'NLL']

# Debugging print statements
print("subset_df head:\n", subset_df.head())
print("subset_df describe:\n", subset_df.describe())
print("subset_df info:\n", subset_df.info())

# Violin plot for the left side
sns.violinplot(x="Model", y="Value", data=subset_df,
               split=True, inner='quart', ax=ax,
               hue="Metric",
               palette={"NLL": 'white'},
               scale='width', width=0.5, linewidth=1,
               bw_adjust=0.35,
               dodge=False,
               linecolor='lightgrey',
               log_scale=True,
               alpha=0.7)

# Scatter plot for data points with error bars
for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    x_positions = np.ones(len(model_data)) * i + 0.4
    ax.scatter(x_positions, model_data['Value'],
               color=color_mapping[model_name], s=4,
               label=model_name)

    # Calculate mean and SEM
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])

    # Plot error bars
    ax.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                color=color_mapping_k[model_name],
                capsize=5, markersize=4)

ax.set_xlabel('')
ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
ax.set_ylabel('Cross-validated NLL')

# Adjust x-tick positions and rotation of labels
current_xticks = ax.get_xticks()
new_xticks = np.array(current_xticks) + 0.4
ax.set_xticks(new_xticks)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust y-axis with a custom scale
def custom_log_formatter(x, pos):
    if x == 1e-5:
        return '0'
    elif x == 1e-5:
        return '0'
    elif x == 1e-3:
        return r'$10^{-3}$'
    else:
        return f'$10^{{{int(np.log10(x))}}}$'

# Adjust y-axis with a custom scale
def custom_log_formatter(x, pos):
    if x == 1e-5:
        return r'$0$'
    elif x == 1e-4:
        return r''
    elif x == 1e-3:
        return r'$10^{-3}$'
    else:
        return f'$10^{{{int(np.log10(x))}}}$'

ax.set_yscale('log')
ax.yaxis.set_major_formatter(FuncFormatter(custom_log_formatter))
ax.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1]))
ax.yaxis.set_minor_locator(LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=12))
ax.yaxis.set_minor_formatter(NullFormatter())

# Remove minor ticks below 0
minor_ticks = [tick for tick in ax.yaxis.get_minor_locator()() if tick > 1e-4]
minor_ticks.append(1e-4)
ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

# Add a white box to "clip" the y-axisaround 10^-3
rect = plt.Rectangle((-0, 10**-4), len(model_names)+1, 0.0001,
                     facecolor="white", edgecolor="none", zorder=10)
ax.add_patch(rect)

# =============================================================================
# # Left diagonal line
# ax.plot((-0.02, 0.04), (-0.02, 0.04), **kwargs)
# # Right diagonal line
# ax.plot((-0.02, 0.02), (-0.02, 0.02), **kwargs)
#
#
# =============================================================================
import matplotlib.transforms as transforms
import matplotlib.patches as patches
# Cover line
kwargs = dict(transform=ax.transAxes, color='white', clip_on=False, lw=6)
ax.plot((-0.015, 0.015), (0.25, 0.28), **kwargs)


# Define the properties for the white rectangle
rect = patches.Rectangle((0.08, 0.6), 0.1, 0.025, color='white', clip_on=False, zorder=3)

# Create a transform to rotate the rectangle by 45 degrees around its center
trans = transforms.Affine2D().rotate_deg_around(0.5, 0.2575, 45) + ax.transAxes

# Apply the transform to the rectangle
rect.set_transform(trans)

# Add the rectangle to the plot
ax.add_patch(rect)

# Add diagonal lines to indicate the break
d = .005  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1, zorder=10)
# Diagonal line on the left
ax.plot((-0.0135, 0.015), (0.23, 0.26), **kwargs)
# Diagonal line on the right
ax.plot((-0.0135, 0.015), (0.27, 0.30), **kwargs)

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add grid
#ax.grid(True, alpha=0.2, which="both")

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value, num_comparisons, heightOffsetScalar=2.2, lengthScalar=0.05):
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

annotate_significance(ax, 3.4, 2.4,
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 5.5, 0.2)
annotate_significance(ax, 3.4, 6.4,
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 5.6, 0.2)

plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1_violin_log_yaxis_break.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
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
    [nll_array_random_p,
     nll_array_bias_p,
     nll_array_win_stay_p,
     nll_array_rw_symm_p,
     nll_array_rw_cond_p,
     nll_array_ck_p,
     nll_array_rwck_p,
     nll_array_delta_p_rw_p]
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
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_rw_cond_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p
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
for rand, bias, wsls, rw, rw_cond, ck, rwck, delta_p_rw, pid in zip(metric[0][0],
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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True, height_ratios=[3,1])

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
ax1.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]))
ax1.yaxis.set_minor_locator(LogLocator(subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=12))
ax1.yaxis.set_minor_formatter(NullFormatter())

minor_ticks = [tick for tick in ax1.yaxis.get_minor_locator()() if tick > 1e-4]
minor_ticks.append(1e-4)
ax1.yaxis.set_minor_locator(FixedLocator(minor_ticks))

rect = plt.Rectangle((-0, 10**-4), len(model_names)+1, 0.0001,
                     facecolor="white", edgecolor="none", zorder=10)
ax1.add_patch(rect)

kwargs = dict(transform=ax1.transAxes, color='white', clip_on=False, lw=6)
ax1.plot((-0.015, 0.015), (0.25, 0.28), **kwargs)

rect = patches.Rectangle((0.08, 0.6), 0.1, 0.025, color='white', clip_on=False, zorder=3)
trans = transforms.Affine2D().rotate_deg_around(0.5, 0.2575, 45) + ax1.transAxes
rect.set_transform(trans)
ax1.add_patch(rect)

d = .005
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, lw=1, zorder=10)
ax1.plot((-0.0135, 0.015), (0.265, 0.295), **kwargs)
ax1.plot((-0.0135, 0.015), (0.225, 0.255), **kwargs)


# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_delta_p_rw_p
diff_rw = nll_array_rw_symm_p - nll_array_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rw = shapiro(diff_rw)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_delta_p_rw_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_delta_p_rw_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rw > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_delta_p_rw_p, nll_array_rw_symm_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_delta_p_rw_p, nll_array_rw_symm_p)
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
                      nll_array_win_stay_p, nll_array_delta_p_rw_p,
                      p_value_winstay, num_comparisons, 20.5, 0.2)
annotate_significance(ax1, 3.4, 7.4,
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 5.6, 0.2)

print('p_value_rwpd', round(p_value_rwpd, 6))
print('p_value_winstay', round(p_value_winstay, 6))

ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

# Second subplot: Histogram
bar_colors = ['blue', 'green', 'red', 'purple', 'cyan', 'pink', 'orange', 'silver']
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

#%% Pairwise model comparison

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# Assuming nll_array_*_p are defined elsewhere
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
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_rw_cond_p, nll_array_ck_p,
    nll_array_rwck_p, nll_array_delta_p_rw_p
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
plt.figure(figsize=(12, 10))
ax = sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='coolwarm',
                 linewidths=0.5, cbar=False,
                 mask=np.isnan(heatmap_data.values),
                 annot_kws={"color": "black"})

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
plt.legend(handles, labels, title="Models", bbox_to_anchor=(1.05, 1.015),
           loc='upper left')

plt.xticks(rotation=45, ha='right')

plt.xlabel('')
plt.ylabel('')
plt.show()



#%% Symlog with Scatter and
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, shapiro, ttest_rel, wilcoxon, sem

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

color_mapping_k = {name: "black" for name in model_names}

# Assuming nll_array_*_p are defined elsewhere
model_values = [
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Remove NaNs and apply log transformation with a small shift
#df['Value'] = df['Value'].replace(0, np.nan).dropna() + 0.01
#df['Value'] = df['Value'] + 0.01
#subset_df = df[df['Metric'] == 'NLL']

# Plot NLL metric
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Custom violin plot using matplotlib
positions = np.arange(len(model_names))
for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]['Value']
    kde = gaussian_kde(model_data, bw_method=0.3)
    x = np.linspace(0, model_data.max(), 400)
    y = kde.evaluate(x)
    y = y / y.max() * 0.3  # Normalize and scale for plotting
    ax.fill_betweenx(x, i - y, i, facecolor='#b4d4df', alpha=1)
    ax.plot(i - y, x, color='#878f92', lw=1)

    # Add box plot lines (quartiles, median, whiskers)
    q1 = np.percentile(model_data, 25)
    q2 = np.percentile(model_data, 50)  # Median
    q3 = np.percentile(model_data, 75)
    iqr = q3 - q1
    lower_whisker = max(model_data.min(), q1 - 1.5 * iqr)
    upper_whisker = min(model_data.max(), q3 + 1.5 * iqr)

    # Plot quartiles
    ax.plot([i - 0.1, i], [q1, q1], color='grey')
    ax.plot([i - 0.1, i], [q3, q3], color='grey')

    # Plot median
    ax.plot([i - 0.2, i], [q2, q2], color='grey', linewidth=1, ls='--')

# =============================================================================
#     # Plot whiskers
#     ax.plot([i - 0.1, i], [lower_whisker, lower_whisker], color='black')
#     ax.plot([i - 0.1, i], [upper_whisker, upper_whisker], color='black')
#     ax.plot([i - 0.1, i - 0.1], [lower_whisker, q1], color='black')
#     ax.plot([i - 0.1, i - 0.1], [q3, upper_whisker], color='black')
# =============================================================================

# Scatter plot for data points with error bars
for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    x_positions = np.ones(len(model_data)) * i + 0.2
    ax.scatter(x_positions, model_data['Value'],
               color=color_mapping[model_name], s=1,
               label=model_name)


    # Calculate mean and SEM
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])

    # Plot error bars
    ax.errorbar(i + 0.2, mean_value, yerr=sem_value, fmt='o',
                color=color_mapping_k[model_name],
                capsize=4, markersize=3)

ax.set_xlabel('')
#ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
ax.set_ylabel('Cross-validated NLL')

# Adjust x-tick positions and rotation of labels
current_xticks = ax.get_xticks()
new_xticks = np.array(current_xticks) + 0.2
ax.set_xticks(new_xticks[1:-1])
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_yscale('symlog')  # Set y-axis to log scale

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value, num_comparisons, heightOffsetScalar=2.2, lengthScalar=0.05):
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

annotate_significance(ax, 3.2, 2.2,
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 3.2, 0.2)
annotate_significance(ax, 3.2, 6.2,
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 3.5, 0.2)

plt.tight_layout()
plt.show()

#%%



#%%  Histogram of best model

metric = [
    [nll_array_random_p,
    nll_array_bias_p,
    nll_array_win_stay_p,
    nll_array_rw_symm_p,
    nll_array_ck_p,
    nll_array_rwck_p,
    nll_array_delta_p_rw_p]
]

# Loop over metrics
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

metric_list = metric[0]
metric_name = 'NLL'

score_board = []
pids = []
# Loop over each participant's model score
for rand, bias, wsls, rw, ck, rwck, delta_p_rw, pid in zip(metric_list[0],
                                                           metric_list[1],
                                                           metric_list[2],
                                                           metric_list[3],
                                                           metric_list[4],
                                                           metric_list[5],
                                                           metric_list[6],
                                                           range(len(metric_list[6]))
                                                           ):

    # Scores from different models
    scores = np.array([rand, bias, wsls, rw, ck, rwck, delta_p_rw])

    # Find the minimum score
    min_score = np.min(scores)

    # Get indices of all occurrences of the lowest score
    # e.g., if multiple models have the lowest score, take both
    idxs = np.where(scores == min_score)[0]

    # Save best models - all models with the lowest score
    for idx in idxs:
        score_board.append(idx)
        pids.append(pid)

# Get pid of participants with RWPD as best model according to NLL
df_best_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
pid_wsls_best = df_best_fit[df_best_fit.best_model==2].pid
pid_rw_best = df_best_fit[df_best_fit.best_model==3].pid
pid_rwpd_best = df_best_fit[df_best_fit.best_model==6].pid

models = ['Random', 'Biased', 'Win-Stay-Loose-Shift', 'RW',
          'Choice Kernel', 'RW + Choice Kernel', 'RW + Performance Delta']
counts = [score_board.count(0), # Random
          score_board.count(1), # Bias
          score_board.count(2), # WSLS
          score_board.count(3), # RW
          score_board.count(4), # CK
          score_board.count(5), # RWCK
          score_board.count(6),] # RWPD

bar_colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
bars = ax.bar(models, counts, color=bar_colors)

# Customizing the Axes
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_ylabel('Best model count')
ax.set_xlim(-1, len(models))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.suptitle('Best Model')
plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_hist_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)
plt.show()

#%%
from scipy.stats import chi2_contingency
import os

def annotate_significance(ax, x1, x2, y_values, p_value, num_comparisons, heightOffsetScalar=1.2, lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot, with Bonferroni correction.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y_values (array-like): The y-values of the groups being compared.
    p_value (float): The p-value from the statistical test.
    num_comparisons (int): The number of comparisons for Bonferroni correction.
    """
    corrected_alpha = 0.05 / num_comparisons  # Bonferroni correction
    y_max = max(y_values) * heightOffsetScalar
    y_min = min(y_values) * 0.9

    if p_value < corrected_alpha / 1000:
        sig_level = '***'
    elif p_value < corrected_alpha / 100:
        sig_level = '**'
    elif p_value < corrected_alpha:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

metric = [
    [nll_array_random_p,
    nll_array_bias_p,
    nll_array_win_stay_p,
    nll_array_rw_symm_p,
    nll_array_ck_p,
    nll_array_rwck_p,
    nll_array_delta_p_rw_p]
]

# Loop over metrics
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

metric_list = metric[0]
metric_name = 'NLL'

score_board = []
pids = []
# Loop over each participant's model score
for rand, bias, wsls, rw, ck, rwck, delta_p_rw, pid in zip(metric_list[0],
                                                           metric_list[1],
                                                           metric_list[2],
                                                           metric_list[3],
                                                           metric_list[4],
                                                           metric_list[5],
                                                           metric_list[6],
                                                           range(len(metric_list[6]))
                                                           ):

    # Scores from different models
    scores = np.array([rand, bias, wsls, rw, ck, rwck, delta_p_rw])

    # Find the minimum score
    min_score = np.min(scores)

    # Get indices of all occurrences of the lowest score
    # e.g., if multiple models have the lowest score, take both
    idxs = np.where(scores == min_score)[0]

    # Save best models - all models with the lowest score
    for idx in idxs:
        score_board.append(idx)
        pids.append(pid)

# Get pid of participants with RWPD as best model according to NLL
df_best_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
pid_wsls_best = df_best_fit[df_best_fit.best_model==2].pid
pid_rw_best = df_best_fit[df_best_fit.best_model==3].pid
pid_rwpd_best = df_best_fit[df_best_fit.best_model==6].pid

models = ['Random', 'Biased', 'Win-Stay-Lose-Shift', 'RW',
          'Choice Kernel', 'RW + Choice Kernel', 'RW + Performance Delta']
counts = [score_board.count(0), # Random
          score_board.count(1), # Bias
          score_board.count(2), # WSLS
          score_board.count(3), # RW
          score_board.count(4), # CK
          score_board.count(5), # RWCK
          score_board.count(6),] # RWPD

bar_colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
bars = ax.bar(models, counts, color=bar_colors)

# Customizing the Axes
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_ylabel('Best model count')
ax.set_xlim(-1, len(models))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Chi-squared test between RW vs WSLS
obs_rw_wsls = [score_board.count(3), score_board.count(2)]
chi2_stat, p_value_rw_wsls, _, _ = chi2_contingency([obs_rw_wsls, [sum(obs_rw_wsls) / 2, sum(obs_rw_wsls) / 2]])

# Chi-squared test between RW vs RWPD
obs_rw_rwpd = [score_board.count(3), score_board.count(6)]
chi2_stat, p_value_rw_rwpd, _, _ = chi2_contingency([obs_rw_rwpd, [sum(obs_rw_rwpd) / 2, sum(obs_rw_rwpd) / 2]])

# Bonferroni correction for multiple comparisons
num_comparisons = 2  # Number of comparisons

# Annotate significance
annotate_significance(ax, 2, 3, counts, p_value_rw_wsls, num_comparisons, heightOffsetScalar=1.1, lengthScalar=0.04)
annotate_significance(ax, 3, 6, counts, p_value_rw_rwpd, num_comparisons, heightOffsetScalar=1.2, lengthScalar=0.035)

#plt.suptitle('Best Model')
plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_hist_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)
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
    x = df_m.alpha_array_delta_p_rw_p[pid_rwpd_best.values]
    y = bdi[pid_rwpd_best.values]
else:
    x = df_m.alpha_array_delta_p_rw_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.scatter(x, y, label='Data points', color='silver',
           s=40, edgecolor='black', linewidths= 0.5,
                      )

# Add the regression line
ax.plot(x, slope * x + intercept,
        color='red',
        label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.66, 1.05),
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
plt.savefig(save_path, bbox_inches='tight', dpi=300)

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
    x = df_m.alpha_array_rw_symm_p[pid_rw_best.values]
    y = bdi[pid_rw_best.values]
else:
    x = df_m.alpha_array_rw_symm_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.scatter(x, y, label='Data points', color='purple',
           s=40, edgecolor='black', linewidths= 0.5,)

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.66, 1.05), xycoords='axes fraction',
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
file_name = 'RW_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% Plot the wsls win boundry vs BDI
import scipy.stats as stats

best_fit = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)
if best_fit:
    x = df_m.win_boundary_WSLS_array_p[pid_wsls_best.values]
    y = bdi[pid_wsls_best.values]
else:
    x = df_m.win_boundary_WSLS_array_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('WSLS win boundary')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)


# Add legend
ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'WSLS_win_boundary_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()


#%%# Function to calculate mean and SEM of the absolute model fit

def calculate_mean_sem(data):
    mean_val = np.mean(data)
    sem_val = np.std(data) / np.sqrt(len(data))
    return mean_val, sem_val

# Dictionary to store the results
results = {}

# List of models and their corresponding metrics in the DataFrame
models_metrics = [
    ['random', 'nll_array_random_p', 'aic_array_random_p', 'bic_array_random_p'],
    ['bias', 'nll_array_bias_p', 'aic_array_bias_p', 'bic_array_bias_p'],
    ['win_stay', 'nll_array_win_stay_p', 'aic_array_win_stay_p', 'bic_array_win_stay_p'],
    #['rw_static', 'nll_array_rw_static_p', 'aic_array_rw_static_p', 'bic_array_rw_static_p'],
    ['rw_symm', 'nll_array_rw_symm_p', 'aic_array_rw_symm_p', 'bic_array_rw_symm_p'],
    #['rw_cond', 'nll_array_rw_cond_p', 'aic_array_rw_cond_p', 'bic_array_rw_cond_p'],
    ['ck', 'nll_array_ck_p', 'aic_array_ck_p', 'bic_array_ck_p'],
    ['rwck', 'nll_array_rwck_p', 'aic_array_rwck_p', 'bic_array_rwck_p'],
    ['delta_p_rw', 'nll_array_delta_p_rw_p', 'aic_array_delta_p_rw_p', 'bic_array_delta_p_rw_p'],
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



#%% Only NLL - Boxplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from scipy.stats import shapiro, ttest_rel, wilcoxon
import os

def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value,
                          num_comparisons, heightOffsetScalar=2.2,
                          lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot with Bonferroni correction.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    num_comparisons (int): The number of comparisons being performed for Bonferroni correction.
    """


    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    # Adjust alpha levels for Bonferroni correction
    if p_value < 0.001 / num_comparisons:
        sig_level = '***'
    elif p_value < 0.01 / num_comparisons:
        sig_level = '**'
    elif p_value < 0.05 / num_comparisons:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

def darken_color(color, amount=0.5):
    """
    Darken a given color.
    """
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex([max(0, min(1, c[i] * (1 - amount))) for i in range(3)])

fig, ax = plt.subplots(figsize=(5, 5))  # Increase the figure size for better readability

model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]
# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # Markers for each model

# Define x-coordinates and offsets for each model within the group
x = np.arange(0, 1)  # Base x-coordinates for metrics
offset = 0.35  # Offset for each model within a group

# Plotting NLL for each model
model_values = [
    nll_array_random_p, nll_array_bias_p,
    nll_array_win_stay_p, nll_array_rw_symm_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[0] + offset * (i - 3)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black'), zorder=0)

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p


_, p_value_normal_winstay = shapiro(diff_winstay)

_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2 # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (-1),
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 1.1, 0.04)

annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (3),
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 1.15, 0.035)

# Customizing the Axes
ax.set_xticks([x[0] + offset * (i - 3) for i in range(len(model_names))])
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.set_ylabel('Cross-validated NLL')
#ax.set_yscale('log')  # Set y-axis to log scale
ax.set_xlim(-1.3, 1.3)
# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

import matplotlib.lines as mlines

# Custom legend handles
custom_handles = [
    mlines.Line2D([], [], color=color, marker='o',
                  linestyle='None', markersize=6,
                  markerfacecolor=color,
                  alpha=1)
    for color in colors
]

# =============================================================================
# # Adding a legend
# ax.legend(custom_handles, model_names, scatterpoints=1, markerscale=1.2,
#           fontsize='small', loc='upper left', bbox_to_anchor=(1, 1))
# =============================================================================

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()

#%% KDE + Boxplot + Scatter + mean+-sem

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}
color_mapping_k = {name: "black" for name in model_names}

# Assuming nll_array_*_p are defined elsewhere
model_values = [
    nll_array_random_p, nll_array_bias_p, nll_array_win_stay_p,
    nll_array_rw_symm_p, nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p
]

data = []
for model_name, nll in zip(model_names, model_values):
    for value in nll:
        data.append([model_name, 'NLL', value])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Plot NLL metric
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

subset_df = df[df['Metric'] == 'NLL']

# Violin plot for the left side
sns.violinplot(x="Model", y="Value", data=subset_df,
               split=True, inner='quart', ax=ax,
               hue="Metric",
               palette={"NLL": 'lightblue'},
               scale='width', width=0.5, linewidth=1,
               bw_adjust=0.35,
               dodge=False,
              # inner_kws = {'whis_width':1,
              #              'box_width': 5,
              #              'marker': '_'}
             )

# =============================================================================
# # Add Boxplot with slight offset to the right
# positions = range(len(model_names))
# offset = 0.25
# positions = [pos + offset for pos in positions]
# sns.boxplot(x="Model", y="Value", data=df, width=0.1, positions=positions,
#             color='lightgrey', fliersize=0, linewidth=1,
#             medianprops=dict(marker='', color='black', markerfacecolor='white',
#                              markersize=5))
# =============================================================================

# Scatter plot for data points with error bars
for i, model_name in enumerate(model_names):
    model_data = subset_df[subset_df['Model'] == model_name]
    x_positions = np.ones(len(model_data)) * i + 0.4
    ax.scatter(x_positions, model_data['Value'],
               color=color_mapping[model_name], s=1,
               label=model_name)

    # Calculate mean and SEM
    mean_value = model_data['Value'].mean()
    sem_value = sem(model_data['Value'])

    # Plot error bars
    ax.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                color=color_mapping_k[model_name],
                capsize=4, markersize=3)

ax.set_xlabel('')
ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
ax.set_ylabel('Cross-validated NLL')


# Adjust x-tick positions and rotation of labels
current_xticks = ax.get_xticks()
new_xticks = np.array(current_xticks) + 0.4
ax.set_xticks(new_xticks)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

# Test for normality
_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
annotate_significance(ax, 3.4, 2.4,
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, num_comparisons, 1.2, 0.05)
annotate_significance(ax, 3.4, 6.4,
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, num_comparisons, 1.3, 0.035)

plt.tight_layout()
plt.show()

#%%  Histogram of best model

metric = [
    [nll_array_random_p,
    nll_array_bias_p,
    nll_array_win_stay_p,
    nll_array_rw_symm_p,
    nll_array_ck_p,
    nll_array_rwck_p,
    nll_array_delta_p_rw_p]
]

# Loop over metrics
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

metric_list = metric[0]
metric_name = 'NLL'

score_board = []
pids = []
# Loop over each participant's model score
for rand, bias, wsls, rw, ck, rwck, delta_p_rw, pid in zip(metric_list[0],
                                                           metric_list[1],
                                                           metric_list[2],
                                                           metric_list[3],
                                                           metric_list[4],
                                                           metric_list[5],
                                                           metric_list[6],
                                                           range(len(metric_list[6]))
                                                           ):

    # Scores from different models
    scores = np.array([rand, bias, wsls, rw, ck, rwck, delta_p_rw])

    # Find the minimum score
    min_score = np.min(scores)

    # Get indices of all occurrences of the lowest score
    # e.g., if multiple models have the lowest score, take both
    idxs = np.where(scores == min_score)[0]

    # Save best models - all models with the lowest score
    for idx in idxs:
        score_board.append(idx)
        pids.append(pid)

# Get pid of participants with RWPD as best model according to NLL
df_best_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
pid_wsls_best = df_best_fit[df_best_fit.best_model==2].pid
pid_rw_best = df_best_fit[df_best_fit.best_model==3].pid
pid_rwpd_best = df_best_fit[df_best_fit.best_model==6].pid

models = ['Random', 'Biased', 'Win-Stay-Loose-Shift', 'RW',
          'Choice Kernel', 'RW + Choice Kernel', 'RW + Performance Delta']
counts = [score_board.count(0), # Random
          score_board.count(1), # Bias
          score_board.count(2), # WSLS
          score_board.count(3), # RW
          score_board.count(4), # CK
          score_board.count(5), # RWCK
          score_board.count(6),] # RWPD

bar_colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
bars = ax.bar(models, counts, color=bar_colors)

# Customizing the Axes
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_ylabel('Best model count')
ax.set_xlim(-1, len(models))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

#plt.suptitle('Best Model')
plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_hist_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)
plt.show()

#%%
from scipy.stats import chi2_contingency
import os

def annotate_significance(ax, x1, x2, y_values, p_value, num_comparisons, heightOffsetScalar=1.2, lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot, with Bonferroni correction.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y_values (array-like): The y-values of the groups being compared.
    p_value (float): The p-value from the statistical test.
    num_comparisons (int): The number of comparisons for Bonferroni correction.
    """
    corrected_alpha = 0.05 / num_comparisons  # Bonferroni correction
    y_max = max(y_values) * heightOffsetScalar
    y_min = min(y_values) * 0.9

    if p_value < corrected_alpha / 1000:
        sig_level = '***'
    elif p_value < corrected_alpha / 100:
        sig_level = '**'
    elif p_value < corrected_alpha:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}

metric = [
    [nll_array_random_p,
    nll_array_bias_p,
    nll_array_win_stay_p,
    nll_array_rw_symm_p,
    nll_array_ck_p,
    nll_array_rwck_p,
    nll_array_delta_p_rw_p]
]

# Loop over metrics
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

metric_list = metric[0]
metric_name = 'NLL'

score_board = []
pids = []
# Loop over each participant's model score
for rand, bias, wsls, rw, ck, rwck, delta_p_rw, pid in zip(metric_list[0],
                                                           metric_list[1],
                                                           metric_list[2],
                                                           metric_list[3],
                                                           metric_list[4],
                                                           metric_list[5],
                                                           metric_list[6],
                                                           range(len(metric_list[6]))
                                                           ):

    # Scores from different models
    scores = np.array([rand, bias, wsls, rw, ck, rwck, delta_p_rw])

    # Find the minimum score
    min_score = np.min(scores)

    # Get indices of all occurrences of the lowest score
    # e.g., if multiple models have the lowest score, take both
    idxs = np.where(scores == min_score)[0]

    # Save best models - all models with the lowest score
    for idx in idxs:
        score_board.append(idx)
        pids.append(pid)

# Get pid of participants with RWPD as best model according to NLL
df_best_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
pid_wsls_best = df_best_fit[df_best_fit.best_model==2].pid
pid_rw_best = df_best_fit[df_best_fit.best_model==3].pid
pid_rwpd_best = df_best_fit[df_best_fit.best_model==6].pid

models = ['Random', 'Biased', 'Win-Stay-Lose-Shift', 'RW',
          'Choice Kernel', 'RW + Choice Kernel', 'RW + Performance Delta']
counts = [score_board.count(0), # Random
          score_board.count(1), # Bias
          score_board.count(2), # WSLS
          score_board.count(3), # RW
          score_board.count(4), # CK
          score_board.count(5), # RWCK
          score_board.count(6),] # RWPD

bar_colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
bars = ax.bar(models, counts, color=bar_colors)

# Customizing the Axes
ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
ax.set_ylabel('Best model count')
ax.set_xlim(-1, len(models))
ax.set_xticklabels(model_names, rotation=45, ha='right')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Chi-squared test between RW vs WSLS
obs_rw_wsls = [score_board.count(3), score_board.count(2)]
chi2_stat, p_value_rw_wsls, _, _ = chi2_contingency([obs_rw_wsls, [sum(obs_rw_wsls) / 2, sum(obs_rw_wsls) / 2]])

# Chi-squared test between RW vs RWPD
obs_rw_rwpd = [score_board.count(3), score_board.count(6)]
chi2_stat, p_value_rw_rwpd, _, _ = chi2_contingency([obs_rw_rwpd, [sum(obs_rw_rwpd) / 2, sum(obs_rw_rwpd) / 2]])

# Bonferroni correction for multiple comparisons
num_comparisons = 2  # Number of comparisons

# Annotate significance
annotate_significance(ax, 2, 3, counts, p_value_rw_wsls, num_comparisons, heightOffsetScalar=1.1, lengthScalar=0.04)
annotate_significance(ax, 3, 6, counts, p_value_rw_rwpd, num_comparisons, heightOffsetScalar=1.2, lengthScalar=0.035)

#plt.suptitle('Best Model')
plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_hist_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)
plt.show()

#%% Plot the alpha of RWPD vs BDI
import scipy.stats as stats

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.alpha_array_delta_p_rw_p[pid_rwpd_best.values]
    y = bdi[pid_rwpd_best.values]
else:
    x = df_m.alpha_array_delta_p_rw_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('RWPD Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)
ax.set_xlim(-0.025, 0.55)

# Add legend
#ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'RWPD_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% Plot the alpha of RW vs BDI
import scipy.stats as stats

best_fits = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique()[:], total=len(df.pid.unique()[:])):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)

if best_fits:
    x = df_m.alpha_array_rw_symm_p[pid_rw_best.values]
    y = bdi[pid_rw_best.values]
else:
    x = df_m.alpha_array_rw_symm_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')
ax.set_xlim(-0.025, 0.55)
# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('RW Learning rate')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)

# Add legend
#ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'RW_alpha_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()

#%% Plot the wsls win boundry vs BDI
import scipy.stats as stats

best_fit = False

# Import data - Fixed feedback condition (Experiment 1)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)
bdi = []
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):

    # Get bdi
    bdi.append(df[df.pid==participant].bdi.unique()[0])

# Extracting the relevant data
bdi = np.array(bdi)
if best_fit:
    x = df_m.win_boundary_WSLS_array_p[pid_wsls_best.values]
    y = bdi[pid_wsls_best.values]
else:
    x = df_m.win_boundary_WSLS_array_p.values
    y = bdi

# Calculate the linear regression and correlation
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Create the scatter plot
fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.scatter(x, y, label='Data points')

# Add the regression line
ax.plot(x, slope * x + intercept, color='red', label='Regression line')

# Annotate with R and p values
annotation_text = f'$R^2 = {r_value**2:.2f}, p = {p_value:.2f}$'
ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=12, ha='left', va='top',
            bbox=dict(facecolor='white', edgecolor='black'))

ax.set_xlabel('WSLS win boundary')
ax.set_ylabel('BDI score')
ax.spines[['top', 'right']].set_visible(False)


# Add legend
ax.legend()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'WSLS_win_boundary_vs_BDI.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save the plot
plt.savefig(save_path, bbox_inches='tight', dpi=300)

# Show the plot
plt.show()


#%% Supplementary

# Whisker plots + scatter

def annotate_significance(ax, x1, x2, y1_values, y2_values, p_value, heightOffsetScalar=2.2, lengthScalar=0.05):
    """
    Annotate significance between two groups on the plot.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to draw the annotations on.
    x1, x2 (float): The x-coordinates of the two groups being compared.
    y1_values, y2_values (array-like): The y-values of the two groups being compared.
    p_value (float): The p-value from the statistical test.
    """
    y_max = max(max(y1_values), max(y2_values)) * heightOffsetScalar
    y_min = min(min(y1_values), min(y2_values)) * 0.9

    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'

    # Draw horizontal line
    ax.plot([x1, x2], [y_max, y_max], color='black')
    # Draw vertical lines
    ax.plot([x1, x1], [y_max - (lengthScalar * y_max), y_max], color='black')
    ax.plot([x2, x2], [y_max - (lengthScalar * y_max), y_max], color='black')
    # Add text
    ax.text((x1 + x2) / 2, y_max, sig_level, ha='center', va='bottom')

fig, ax = plt.subplots(figsize=(10, 6))  # Increase the figure size for better readability

# Define x-coordinates and offsets for each model within the group
x = np.arange(0, 9, 3)  # Base x-coordinates for metrics
offset = 0.35  # Offset for each model within a group

# Colors for each model (added one for RW symmetric LR model)
colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
markers = ['o', 'o', 'o', 'o', 'o', 'o', 'o']  # Different markers for each model

# Plotting NLL for each model
model_values = [
    nll_array_random_p, nll_array_bias_p,
    nll_array_win_stay_p, nll_array_rw_symm_p,
    nll_array_ck_p, nll_array_rwck_p,
    nll_array_delta_p_rw_p,
]
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW Symmetric LR", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

scatter_handles = []
box_handles = []
for i, values in enumerate(model_values):
    # Box plot
    _handle = ax.boxplot(
        values, positions=[x[0] + offset * (i - 3)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black')
    )
    box_handles.append(_handle["boxes"][0])

# =============================================================================
#     # Scatter plot of individual samples
#     scatter_handle = ax.scatter(
#         np.full(values.shape, x[0] + offset * (i - 3)),
#         values, color=colors[i], alpha=0.7, s=5, marker=markers[i], label=model_names[i]
#     )
#     scatter_handles.append(scatter_handle)
# =============================================================================

# Perform normality test for the differences
diff_winstay = nll_array_rw_symm_p - nll_array_win_stay_p
diff_rwpd = nll_array_rw_symm_p - nll_array_delta_p_rw_p

_, p_value_normal_winstay = shapiro(diff_winstay)
_, p_value_normal_rwpd = shapiro(diff_rwpd)

# Bonferroni correction
num_comparisons = 2  # Two comparisons: RW vs. WSLS and RW vs. RWPD
alpha = 0.05 / num_comparisons

# Choose the test based on the normality of the differences
if p_value_normal_winstay > alpha:
    _, p_value_winstay = ttest_rel(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('paired t-test RW and WSLS')
else:
    _, p_value_winstay = wilcoxon(nll_array_rw_symm_p, nll_array_win_stay_p)
    print('wilcoxon RW and WSLS')

if p_value_normal_rwpd > alpha:
    _, p_value_rwpd = ttest_rel(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('paired t-test RW and RWPD')
else:
    _, p_value_rwpd = wilcoxon(nll_array_rw_symm_p, nll_array_delta_p_rw_p)
    print('wilcoxon RW and RWPD')

# Annotate significance
annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (-1),
                      nll_array_rw_symm_p, nll_array_win_stay_p,
                      p_value_winstay, 1.2, 0.05)
annotate_significance(ax, x[0] + offset * (0), x[0] + offset * (3),
                      nll_array_rw_symm_p, nll_array_delta_p_rw_p,
                      p_value_rwpd, 1.3, 0.035)

# Plotting AIC for each model
model_values = [
    aic_array_random_p, aic_array_bias_p,
    aic_array_win_stay_p, aic_array_rw_symm_p,
    aic_array_ck_p, aic_array_rwck_p,
    aic_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[1] + offset * (i - 3)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black')
    )
# =============================================================================
#     # Scatter plot of individual samples
#     scatter_handle = ax.scatter(
#         np.full(values.shape, x[1] + offset * (i - 3)),
#         values, color=colors[i], alpha=0.7, s=20, marker=markers[i], label='_Hidden label'
#     )
#     scatter_handles.append(scatter_handle)
#
# =============================================================================
# Plotting BIC for each model
model_values = [
    bic_array_random_p, bic_array_bias_p,
    bic_array_win_stay_p, bic_array_rw_symm_p,
    bic_array_ck_p, bic_array_rwck_p,
    bic_array_delta_p_rw_p,
]

for i, values in enumerate(model_values):
    # Box plot
    ax.boxplot(
        values, positions=[x[2] + offset * (i - 3)],
        widths=0.2, patch_artist=True,
        boxprops=dict(facecolor=colors[i], color='black'),
        medianprops=dict(color='black')
    )
# =============================================================================
#     # Scatter plot of individual samples
#     scatter_handle = ax.scatter(
#         np.full(values.shape, x[2] + offset * (i - 3)),
#         values, color=colors[i], alpha=0.7, s=20, marker=markers[i], label='_Hidden label'
#     )
#     scatter_handles.append(scatter_handle)
# =============================================================================

# Customizing the Axes
ax.set_xticks(x)
ax.set_xticklabels(['NLL', 'AIC', 'BIC'])
ax.set_ylabel('Model fits')
#ax.set_yscale('log')  # Set y-axis to log scale

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# =============================================================================
# # Adding a legend
# ax.legend(handles=scatter_handles[:len(model_names)], labels=model_names,
#           scatterpoints=1, markerscale=2, fontsize='medium', title='')
#
# =============================================================================
# Adding a legend for boxplots
box_labels = [model_names[i] for i in range(len(model_names))]
ax.legend(handles=box_handles, labels=box_labels, loc='upper left', bbox_to_anchor=(0.025, 1), fontsize='small')


# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_NLL_means_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models', result_path, file_name)

# Save
plt.savefig(save_path, bbox_inches='tight', dpi=300)

plt.tight_layout()
plt.show()

#%% KDE + Boxplot + Scatter + mean+-sem

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}
color_mapping_k = {name: "black" for name in model_names}

model_values = [
    (nll_array_random_p, aic_array_random_p, bic_array_random_p),
    (nll_array_bias_p, aic_array_bias_p, bic_array_bias_p),
    (nll_array_win_stay_p, aic_array_win_stay_p, bic_array_win_stay_p),
    (nll_array_rw_symm_p, aic_array_rw_symm_p, bic_array_rw_symm_p),
    (nll_array_ck_p, aic_array_ck_p, bic_array_ck_p),
    (nll_array_rwck_p, aic_array_rwck_p, bic_array_rwck_p),
    (nll_array_delta_p_rw_p, aic_array_delta_p_rw_p, bic_array_delta_p_rw_p),
]

data = []
for model_name, values in zip(model_names, model_values):
    for nll, aic, bic in zip(*values):
        data.append([model_name, 'NLL', nll])
        data.append([model_name, 'AIC', aic])
        data.append([model_name, 'BIC', bic])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Plot violin plots
fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)

metrics = ['NLL', 'AIC', 'BIC']
for ax, metric in zip(axes, metrics):
    subset_df = df[df['Metric'] == metric]

    # Violin plot for the left side
    sns.violinplot(x="Model", y="Value", data=subset_df,
                   split=True, inner='quart', ax=ax,
                   hue="Metric",
                   palette={"NLL": 'lightblue',
                            "AIC": 'lightblue',
                            "BIC": 'lightblue'},
                   scale='width', width=0.5, linewidth=1,
                   bw_adjust=0.35,
                   dodge=False)

    # Scatter plot for data points with error bars
    for i, model_name in enumerate(model_names):
        model_data = subset_df[subset_df['Model'] == model_name]
        x_positions = np.ones(len(model_data)) * i + 0.4
        ax.scatter(x_positions, model_data['Value'],
                   color=color_mapping[model_name], s=1,
                   label=model_name)

        # Calculate mean and SEM
        mean_value = model_data['Value'].mean()
        sem_value = sem(model_data['Value'])

        # Plot error bars
        ax.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                    color=color_mapping_k[model_name],
                    capsize=4, markersize=3)

    ax.set_title(metric)
    ax.set_xlabel('')
    ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
axes[0].set_ylabel('Model Fit')

# Rotate x-axis labels
for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Remove top and right spines
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()

plt.savefig('test_fig.png', dpi=300)
plt.show()


#%%

# Create a dataframe
model_names = [
    "Random", "Biased", "Win-Stay-Lose-Shift",
    "RW", "Choice Kernel",
    "RW + Choice Kernel", "RW + Performance Delta",
]

# Define color mapping using the approximate colors
color_mapping = {
    "Random": '#0000ff',
    "Biased": '#008000',
    "Win-Stay-Lose-Shift": '#ff0000',
    "RW": '#800080',
    "Choice Kernel": '#ffc0cb',
    "RW + Choice Kernel": '#ffa500',
    "RW + Performance Delta": '#c0c0c0'
}
color_mapping_k = {name: "black" for name in model_names}

model_values = [
    (nll_array_random_p, aic_array_random_p, bic_array_random_p),
    (nll_array_bias_p, aic_array_bias_p, bic_array_bias_p),
    (nll_array_win_stay_p, aic_array_win_stay_p, bic_array_win_stay_p),
    (nll_array_rw_symm_p, aic_array_rw_symm_p, bic_array_rw_symm_p),
    (nll_array_ck_p, aic_array_ck_p, bic_array_ck_p),
    (nll_array_rwck_p, aic_array_rwck_p, bic_array_rwck_p),
    (nll_array_delta_p_rw_p, aic_array_delta_p_rw_p, bic_array_delta_p_rw_p),
]

data = []
for model_name, values in zip(model_names, model_values):
    for nll, aic, bic in zip(*values):
        data.append([model_name, 'NLL', nll])
        data.append([model_name, 'AIC', aic])
        data.append([model_name, 'BIC', bic])

df = pd.DataFrame(data, columns=["Model", "Metric", "Value"])

# Plot violin plots
fig, axes = plt.subplots(3, 1, figsize=(5, 10), sharex=True)

metrics = ['NLL', 'AIC', 'BIC']
for ax, metric in zip(axes, metrics):
    subset_df = df[df['Metric'] == metric]

    # Violin plot for the left side
    sns.violinplot(x="Model", y="Value", data=subset_df,
                   split=True, inner='quart', ax=ax,
                   hue="Metric",
                   palette={"NLL": 'lightblue',
                            "AIC": 'lightblue',
                            "BIC": 'lightblue'},
                   scale='width', width=0.5, linewidth=1,
                   bw_adjust=0.35,
                   dodge=False)

    # Scatter plot for data points with error bars
    for i, model_name in enumerate(model_names):
        model_data = subset_df[subset_df['Model'] == model_name]
        x_positions = np.ones(len(model_data)) * i + 0.4
        ax.scatter(x_positions, model_data['Value'],
                   color=color_mapping[model_name], s=2,
                   label=model_name)

        # Calculate mean and SEM
        mean_value = model_data['Value'].mean()
        sem_value = sem(model_data['Value'])

        # Plot error bars
        ax.errorbar(i + 0.4, mean_value, yerr=sem_value, fmt='o',
                    color=color_mapping_k[model_name],
                    capsize=5, markersize=2)

    #ax.set_title(metric)
    ax.set_xlabel('')
    ax.get_legend().remove()  # Remove legend created by scatter plot

# Customizing the Axes
axes[0].set_ylabel('NLL')
axes[1].set_ylabel('AIC')
axes[2].set_ylabel('BIC')

# Rotate x-axis labels
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')

# Remove top and right spines
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.show()

#%%  Histogram of best model

metric = [
    [nll_array_random_p,
    nll_array_bias_p,
    nll_array_win_stay_p,
    nll_array_rw_symm_p,
    nll_array_ck_p,
    nll_array_rwck_p,
    nll_array_delta_p_rw_p,],

    [aic_array_random_p,
    aic_array_bias_p,
    aic_array_win_stay_p,
    aic_array_rw_symm_p,
    aic_array_ck_p,
    aic_array_rwck_p,
    aic_array_delta_p_rw_p],

    [bic_array_random_p,
    bic_array_bias_p,
    bic_array_win_stay_p,
    bic_array_rw_symm_p,
    bic_array_ck_p,
    bic_array_rwck_p,
    bic_array_delta_p_rw_p]

          ]

# Loop over metrics
fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize=(6,6))
for metric_list, metric_name, ax in zip(metric,
                                        ['NLL', 'AIC', 'BIC'],
                                        [ax, ax2, ax3]):

    score_board = []
    pids = []
    # Loop over each participant's model score
    for rand, bias, wsls, rw, ck, rwck, delta_p_rw, pid in zip(metric_list[0],
                                                               metric_list[1],
                                                               metric_list[2],
                                                               metric_list[3],
                                                               metric_list[4],
                                                               metric_list[5],
                                                               metric_list[6],
                                                     range(len(metric_list[6]))
                                                               ):

        # Scores from different models
        scores = np.array([rand, bias, wsls, rw, ck, rwck, delta_p_rw])

        # Find the minimum score
        min_score = np.min(scores)

        # Get indices of all occurrences of the lowest score
        # e.g., if multiple models have the lowest score, take both
        idxs = np.where(scores == min_score)[0]

        # Save best models - all models with the lowest score
        for idx in idxs:
            score_board.append(idx)
            pids.append(pid)

    # Get pid of participants with RWPD as best model according to NLL
    if metric_name == 'NLL':
        df_best_fit = pd.DataFrame({'pid': pids, 'best_model': score_board})
        pid_wsls_best = df_best_fit[df_best_fit.best_model==2].pid
        pid_rw_best = df_best_fit[df_best_fit.best_model==3].pid
        pid_rwpd_best = df_best_fit[df_best_fit.best_model==6].pid
    else:
        pass

    models = ['random', 'biased', 'WSLS', 'RW',
              'CK', 'RWCK', 'RWPD']
    counts = [score_board.count(0), # Rand
              score_board.count(1), # Bias
              score_board.count(2), # WSLS
              score_board.count(3), # RW
              score_board.count(4), # CK
              score_board.count(5), # RWCK
              score_board.count(6),] #RWPD

    bar_colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']
    bars = ax.bar(models, counts, color=bar_colors)

    # Customizing the Axes
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_ylabel('Best Model Count')
    ax.set_xlim(-1, len(models))
    ax.set_title(metric_name)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()

# Set save path
result_path = r"results\Fixed_feedback\model_comparison"
file_name = 'model_comparison_hist_CV1.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, file_name)

# Save
plt.savefig(save_path,
            bbox_inches='tight',
            dpi=300)
plt.show()


