# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 00:32:07 2025

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


# Load data
df_a = load_df(EXP=1)

# Add session column
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Compute absolute error per subtrial
df['difference'] = abs(df['estimate'] - df['correct'])

# Compute mean absolute error per (pid, trial, session)
abs_error = df.groupby(['pid', 'trial', 'session'])['difference'].mean().reset_index(name='abs_error')

# Merge back into the full df (so each subtrial gets the abs_error for its trial)
df = df.merge(abs_error, on=['pid', 'trial', 'session'], how='left')

# Keep only the first subtrial row
df_trial_level = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')

# Sort by pid, session, and trial to ensure correct order
df_trial_level = df_trial_level.sort_values(by=['pid', 'session', 'trial'])

# Add feedback from the previous trial
df_trial_level['feedback_prev'] = df_trial_level.groupby(['pid', 'session'])['feedback'].shift(1)

# Fill NaN feedback_prev with -1 or another neutral value
df_trial_level['feedback_prev'] = df_trial_level['feedback_prev'].fillna(-1)

# Drop rows with missing feedback_prev or confidence
df_lmm = df_trial_level.dropna(subset=['confidence', 'abs_error', 'feedback_prev'])

model = smf.mixedlm(
    "confidence ~ abs_error + feedback_prev",
    data=df_lmm,
    groups=df_lmm["pid"],
    re_formula="~1"
)

result = model.fit()

# Print summary of the results
print(result.summary())

#%% Vizualise model fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Build prediction grid for abs_error and feedback_prev
abs_error_vals = np.linspace(df_lmm['abs_error'].min(), df_lmm['abs_error'].max(), 50)
feedback_vals = np.linspace(df_lmm['feedback_prev'].min(), df_lmm['feedback_prev'].max(), 50)

grid = pd.DataFrame([(a, f) for a in abs_error_vals for f in feedback_vals],
                    columns=['abs_error', 'feedback_prev'])

# Add a dummy pid (any value from your data)
grid['pid'] = df_lmm['pid'].iloc[0]

# Predict confidence using the fitted model
grid['predicted_confidence'] = result.predict(grid)

# Scatter of observed data (with color mapped to confidence)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_lmm, x='abs_error', y='feedback_prev', hue='confidence',
                palette='coolwarm', alpha=0.5, edgecolor=None)

# Contour of predicted confidence
contour = grid.pivot(index='feedback_prev', columns='abs_error', values='predicted_confidence')
X, Y = np.meshgrid(abs_error_vals, feedback_vals)
plt.contour(X, Y, contour.values, levels=20, cmap='coolwarm', alpha=0.7)

# Heatmap background (optional alternative to contour)
plt.imshow(contour.values, extent=[abs_error_vals.min(), abs_error_vals.max(),
              feedback_vals.min(), feedback_vals.max()],
              origin='lower', aspect='auto', cmap='coolwarm', alpha=0.2)

plt.colorbar(label='Predicted Confidence')
plt.title('Model Prediction Surface: Confidence ~ abs_error + feedback_prev')
plt.xlabel('Absolute Error')
plt.ylabel('Previous Feedback')
plt.tight_layout()
plt.show()

#%% Calculate metacognitive bias (intercept) vs bdi score

import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np

plt.rcParams.update({
    'font.size': 14,             # Global font size
    'axes.titlesize': 14,        # Title size
    'axes.labelsize': 14,        # Axis label size
    'xtick.labelsize': 14,       # X tick label size
    'ytick.labelsize': 14,       # Y tick label size
    'legend.fontsize': 14,       # Legend text
    'legend.title_fontsize': 14  # Legend title
})

# Load and process data
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)
df['difference'] = abs(df['estimate'] - df['correct'])

abs_error = df.groupby(['pid', 'trial', 'session'])['difference'].mean().reset_index(name='abs_error')
df = df.merge(abs_error, on=['pid', 'trial', 'session'], how='left')

df_trial_level = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial_level = df_trial_level.sort_values(by=['pid', 'session', 'trial'])

# Fit random intercept model
model = smf.mixedlm(
    "confidence ~ abs_error",
    data=df_lmm,
    groups=df_lmm["pid"],
    re_formula="~1"
)

result = model.fit()
print(result.summary())

# Extract random intercepts per participant
random_effects = result.random_effects
intercepts_df = pd.DataFrame({
    "pid": list(random_effects.keys()),
    "intercept": [result.fe_params["Intercept"] + v["Group"] for v in random_effects.values()]
})

# Merge with bdi_score
bdi_scores = df_lmm[['pid', 'bdi']].drop_duplicates()
intercepts_df = intercepts_df.merge(bdi_scores, on='pid', how='left')

# Fit linear regression using statsmodels
reg_result = smf.ols("intercept ~ bdi", data=intercepts_df).fit()
slope = reg_result.params["bdi"]
p_value = reg_result.pvalues["bdi"]
r_squared = reg_result.rsquared

# Plot with regression line
fig, ax = plt.subplots(1,1, figsize=(4,4))
sns.regplot(
    data=intercepts_df,
    x="bdi", y="intercept",
    ci=None,
    color='tab:blue',
    scatter_kws={"s": 40, "alpha": 0.5}
)
# Axis labels
ax.set_xlabel("BDI Score", labelpad=6)
ax.set_ylabel("Metacognitive Bias", labelpad=6)

# Add slope, R², and p-value to plot
stats_text = f"Slope = {slope:.2f}\n$R^2$ = {r_squared:.2f}\n$p$ = {p_value:.3f}"
plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# Remove spines
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

save_folder = r"C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/analysis"
file_name = 'metacognitive_bias'
save_path = os.path.join(save_folder, file_name)
fig.savefig(f'{save_path}.png', dpi=300)
plt.show()

#%% Is BDI a predictor of condition effect size?

# Load and process data
df_a = load_df(EXP=1)

# Add session column
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Calculate abs error
df['difference'] = abs(df['estimate'] - df['correct'])
abs_error = df.groupby(['pid', 'trial', 'session'])['difference'].mean().reset_index(name='abs_error')
df = df.merge(abs_error, on=['pid', 'trial', 'session'], how='left')

# Remove subtrials, keep first
df_trial_level = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial_level = df_trial_level.sort_values(by=['pid', 'session', 'trial'])

# Count how many unique sessions each participant completed
session_counts = df_trial_level.groupby('pid')['session'].nunique()

# Only keep participants who have exactly 3 sessions
pids_with_all_sessions = session_counts[session_counts == 3].index

# Filter the main DataFrame
df_all_sessions = df_trial_level[df_trial_level['pid'].isin(pids_with_all_sessions)]

# Calculate the the mean confidence of the 5 last trial per participant-session
# combo.

# Calculate the difference between neut vs pos and neut vs neg conditions
# in the end_of_trial_bin. This will be the effect size of condition on
# confidence.

# Run a linear mixed model predicting the condition effect size from bdi_score.
# Use pid as random variable, with flexible intercept.

# Print summary

# Plot data, regression line,


#%% Is BDI a predictor of condition effect size?

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load and process data
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Remove baseline trials before analysis
df = df[df['condition'] != 'baseline'].copy()

# Calculate abs error per subtrial
df['difference'] = abs(df['estimate'] - df['correct'])
abs_error = df.groupby(['pid', 'trial', 'session'])['difference'].mean().reset_index(name='abs_error')
df = df.merge(abs_error, on=['pid', 'trial', 'session'], how='left')

# Remove subtrials, keep first
df_trial = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])

# Only keep participants with all 3 sessions
session_counts = df_trial.groupby('pid')['session'].nunique()
valid_pids = session_counts[session_counts == 3].index
df_trial = df_trial[df_trial['pid'].isin(valid_pids)]

# Mean confidence of last 5 trials per participant-session
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])  # if not already sorted
chosen_trials = df_trial.groupby(['pid', 'session']).tail(5)

# Mean confidence per condition per participant
confidence_summary = (
    chosen_trials.groupby(['pid', 'condition'])['confidence']
    .mean()
    .reset_index()
    .pivot(index='pid', columns='condition', values='confidence')
    .dropna()
    .reset_index()
)

# Compute confidence differences (vs. neut)
confidence_summary['confidence_diff_pos_vs_neut'] = confidence_summary['pos'] - confidence_summary['neut']
confidence_summary['confidence_diff_neg_vs_neut'] = confidence_summary['neg'] - confidence_summary['neut']

# Add BDI
bdi_scores = df_trial[['pid', 'bdi']].drop_duplicates(subset='pid')
confidence_summary = confidence_summary.merge(bdi_scores, on='pid', how='left')

# Long-form for regression + plotting
long_df = pd.melt(confidence_summary,
                  id_vars=['pid', 'bdi'],
                  value_vars=['confidence_diff_pos_vs_neut', 'confidence_diff_neg_vs_neut'],
                  var_name='comparison',
                  value_name='confidence_difference')

# Run one OLS regression per comparison
ols_results = {}
for comp in ['confidence_diff_pos_vs_neut', 'confidence_diff_neg_vs_neut']:
    df_sub = long_df[long_df['comparison'] == comp]

    print(f"\n========= OLS for {comp.replace('_', ' ')} ============\n")
    model = smf.ols("confidence_difference ~ bdi", df_sub)
    result = model.fit()
    print(result.summary())

    ols_results[comp] = (df_sub, result)

# Plot results
colors = {'confidence_diff_pos_vs_neut': 'tab:blue',
          'confidence_diff_neg_vs_neut': 'tab:red'}

fig, ax = plt.subplots(figsize=(7, 5))

for comp, (df_sub, result) in ols_results.items():
    x = df_sub['bdi']
    y = df_sub['confidence_difference']
    color = colors[comp]
    label = comp.replace('_', ' ').title()

    ax.scatter(x, y, alpha=0.6, label=label, color=color)

    # Regression line
    x_vals = pd.Series(sorted(x))
    y_pred = result.predict(exog=dict(bdi=x_vals))
    ax.plot(x_vals, y_pred, label=f"{label} (fit)", color=color)

ax.set_title("BDI Predicts Confidence Differences Between Conditions", fontsize=14)
ax.set_xlabel("BDI Score")
ax.set_ylabel("Confidence Difference (vs. Neut)")
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()


#%% LMM
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load and process data
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Compute absolute error
df['difference'] = abs(df['estimate'] - df['correct'])
abs_error = df.groupby(['pid', 'trial', 'session'])['difference'].mean().reset_index(name='abs_error')
df = df.merge(abs_error, on=['pid', 'trial', 'session'], how='left')

# Trial-level: remove subtrials
df_trial = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])

# Keep only participants with all 3 sessions
session_counts = df_trial.groupby('pid')['session'].nunique()
valid_pids = session_counts[session_counts == 3].index
df_trial = df_trial[df_trial['pid'].isin(valid_pids)]

# Get last 5 trials per session
last_trials = df_trial.groupby(['pid', 'session']).tail(20).copy()

# Recode condition as categorical with 'neut' as baseline
last_trials['condition'] = pd.Categorical(last_trials['condition'], categories=['neut', 'pos', 'neg'])

# Drop rows with missing model-relevant variables
last_trials = last_trials.dropna(subset=['confidence', 'condition', 'bdi', 'pid'])

# Ensure condition is categorical
last_trials['condition'] = pd.Categorical(
    last_trials['condition'],
    categories=['neut', 'pos', 'neg'],
    ordered=False
)

# Fit LMM: confidence ~ condition * bdi + (1 | pid)
model = smf.mixedlm("confidence ~ condition * bdi", last_trials, groups=last_trials["pid"])
result = model.fit()
print(result.summary())

# Plot estimated confidence by BDI for each condition
colors = {'neut': 'gray', 'pos': 'tab:blue', 'neg': 'tab:red'}
fig, ax = plt.subplots(figsize=(7, 5))

for condition in ['neut', 'pos', 'neg']:
    df_sub = last_trials[last_trials['condition'] == condition]
    x_vals = pd.Series(sorted(df_sub['bdi'].unique()))
    pred_df = pd.DataFrame({'bdi': x_vals, 'condition': condition})
    y_pred = result.predict(pred_df)

    ax.plot(x_vals, y_pred, label=condition.title(), color=colors[condition])
    ax.scatter(df_sub['bdi'], df_sub['confidence'], alpha=0.3, color=colors[condition])

ax.set_title("Confidence by Condition × BDI", fontsize=14)
ax.set_xlabel("BDI Score")
ax.set_ylabel("Confidence (last 5 trials)")
ax.legend(title="Condition")
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

#%% Feedback - Baseline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load data and preprocess
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Remove subtrials and sort
df_trial = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])

# Only keep participants with all 3 feedback conditions
feedback_sessions = df_trial[df_trial['condition'] != 'baseline']
session_counts = feedback_sessions.groupby('pid')['condition'].nunique()
valid_pids = session_counts[session_counts == 3].index

df_trial = df_trial[df_trial['pid'].isin(valid_pids)]

# Split into baseline and feedback trials
baseline_df = df_trial[df_trial['condition'] == 'baseline']
feedback_df = df_trial[df_trial['condition'] != 'baseline']

# Get mean confidence during baseline for each pid-session
baseline_means = (
    baseline_df.groupby(['pid', 'session'])['confidence']
    .mean()
    .reset_index()
    .rename(columns={'confidence': 'baseline_confidence'})
)

# Get mean confidence of last 5 feedback trials for each pid-session
last_feedback_trials = feedback_df.groupby(['pid', 'session']).tail(5)
feedback_means = (
    last_feedback_trials.groupby(['pid', 'session', 'condition'])['confidence']
    .mean()
    .reset_index()
    .rename(columns={'confidence': 'feedback_confidence'})
)

# Merge baseline into feedback to compute within-session effect
merged = feedback_means.merge(baseline_means, on=['pid', 'session'], how='left')
merged['confidence_effect'] = merged['feedback_confidence'] - merged['baseline_confidence']

# Pivot to get one row per pid with condition-specific effects
effects = merged.pivot(index='pid', columns='condition', values='confidence_effect').reset_index()

# Compute condition difference scores (same logic as before)
effects['neut_vs_neg'] = effects['neut'] - effects['neg']
effects['neut_vs_pos'] = effects['neut'] - effects['pos']

# Add BDI
bdi = df_trial[['pid', 'bdi']].drop_duplicates(subset='pid')
effects = effects.merge(bdi, on='pid', how='left')

# Melt for long-form regression
long_df = pd.melt(
    effects,
    id_vars=['pid', 'bdi'],
    value_vars=['neut_vs_neg', 'neut_vs_pos'],
    var_name='comparison',
    value_name='confidence_difference'
)

# Run OLS per comparison
ols_results = {}
for comp in ['neut_vs_neg', 'neut_vs_pos']:
    df_sub = long_df[long_df['comparison'] == comp]

    print(f"\n========= OLS for {comp.replace('_', ' ')} ============\n")
    model = smf.ols("confidence_difference ~ bdi", df_sub)
    result = model.fit()
    print(result.summary())

    ols_results[comp] = (df_sub, result)

# Plot results
colors = {'neut_vs_neg': 'tab:red', 'neut_vs_pos': 'tab:blue'}
fig, ax = plt.subplots(figsize=(7, 5))

for comp, (df_sub, result) in ols_results.items():
    x = df_sub['bdi']
    y = df_sub['confidence_difference']
    color = colors[comp]
    label = comp.replace('_', ' ').title()

    ax.scatter(x, y, alpha=0.6, label=label, color=color)

    # Regression line
    x_vals = pd.Series(sorted(x))
    y_pred = result.predict(exog=dict(bdi=x_vals))
    ax.plot(x_vals, y_pred, label=f"{label} (fit)", color=color)

ax.set_title("BDI Predicts Condition-Specific Confidence Differences", fontsize=14)
ax.set_xlabel("BDI Score")
ax.set_ylabel("Confidence Difference (Feedback - Baseline)")
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

#%%
"""
Does BDI predict how much confidence increases (or decreases) after feedback,
relative to baseline, in each condition?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load data and preprocess
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Remove subtrials and sort
df_trial = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])

# Filter for participants who completed all 3 feedback conditions
feedback_df = df_trial[df_trial['condition'] != 'baseline']
session_counts = feedback_df.groupby('pid')['condition'].nunique()
valid_pids = session_counts[session_counts == 3].index
df_trial = df_trial[df_trial['pid'].isin(valid_pids)]

# Separate baseline and feedback trials
baseline_df = df_trial[df_trial['condition'] == 'baseline']
feedback_df = df_trial[df_trial['condition'] != 'baseline']

# Get mean confidence during baseline per pid-session
baseline_means = (
    baseline_df.groupby(['pid', 'session'])['confidence']
    .mean()
    .reset_index()
    .rename(columns={'confidence': 'baseline_confidence'})
)

# Get mean confidence of last 5 feedback trials per pid-session
last_feedback = feedback_df.groupby(['pid', 'session']).tail(5)
feedback_means = (
    last_feedback.groupby(['pid', 'session', 'condition'])['confidence']
    .mean()
    .reset_index()
    .rename(columns={'confidence': 'feedback_confidence'})
)

# Merge and compute condition-specific confidence effect (feedback - baseline)
merged = feedback_means.merge(baseline_means, on=['pid', 'session'], how='left')
merged['confidence_difference'] = merged['feedback_confidence'] - merged['baseline_confidence']

# Add BDI scores
bdi = df_trial[['pid', 'bdi']].drop_duplicates(subset='pid')
merged = merged.merge(bdi, on='pid', how='left')

# Add metacognitive bias
merged = merged.merge(intercepts_df, on='pid', how='left')

# Recode condition for plot clarity
merged['condition'] = pd.Categorical(merged['condition'], categories=['neut', 'pos', 'neg'])

# Fit one regression per condition
ols_results = {}
for cond in ['neut', 'pos', 'neg']:
    df_sub = merged[merged['condition'] == cond]

    print(f"\n========= OLS for {cond} (Feedback - Baseline) ============\n")
    model = smf.ols("confidence_difference ~ bdi", df_sub)
    result = model.fit()
    print(result.summary())

    ols_results[cond] = (df_sub, result)

# Plot
colors = {'neut': 'gray', 'pos': 'green', 'neg': 'red'}
fig, ax = plt.subplots(figsize=(4.5, 4))

for cond, (df_sub, result) in ols_results.items():
    x = df_sub['bdi']
    y = df_sub['confidence_difference']
    color = colors[cond]
    label = cond.title()

    # Scatter plot
    ax.scatter(x, y, alpha=0.6, label=label, color=color, s=40)

    # Regression line
    x_vals = pd.Series(sorted(x))
    y_pred = result.predict(exog=dict(bdi=x_vals))
    ax.plot(x_vals, y_pred, color=color, linewidth=2)

    # Extract p-value for bdi
    p_val = result.pvalues.get("bdi", np.nan)

    # Annotate with "n.s." if not significant
    if pd.notnull(p_val) and p_val > 0.05:
        # Position label at the end of the line
        x_annot = x_vals.max()-4
        y_annot = y_pred.iloc[-1]
        ax.text(x_annot, y_annot, "n.s.",
                color='k', fontsize=12,
                verticalalignment='bottom',
                horizontalalignment='left')

# Axis formatting
ax.set_xlabel("BDI Score", fontsize=16)
ax.set_ylabel("ΔConfidence\nBaseline - Feedback", fontsize=16)
ax.tick_params(axis='both', labelsize=16)

# Legend and layout
ax.legend(title_fontsize=11, fontsize=14,
          loc='lower right',)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

save_folder = r"C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/analysis"
file_name = 'Confidence_change_vs_BDI'
save_path = os.path.join(save_folder, file_name)
fig.savefig(f'{save_path}.png', dpi=300)

plt.show()

#%% BDI predict Delta Abs Error (baseline - feedback)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load and process data
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Compute absolute error
df['difference'] = abs(df['estimate'] - df['correct'])
abs_error = df.groupby(['pid', 'trial', 'session'])['difference'].mean().reset_index(name='abs_error')
df = df.merge(abs_error, on=['pid', 'trial', 'session'], how='left')

# Remove subtrials and sort
df_trial = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])

# Keep only participants with all 3 feedback conditions
feedback_df = df_trial[df_trial['condition'] != 'baseline']
session_counts = feedback_df.groupby('pid')['condition'].nunique()
valid_pids = session_counts[session_counts == 3].index
df_trial = df_trial[df_trial['pid'].isin(valid_pids)]

# Split baseline and feedback
baseline_df = df_trial[df_trial['condition'] == 'baseline']
feedback_df = df_trial[df_trial['condition'] != 'baseline']

# Mean abs error during baseline per pid-session
baseline_means = (
    baseline_df.groupby(['pid', 'session'])['abs_error']
    .mean()
    .reset_index()
    .rename(columns={'abs_error': 'baseline_abs_error'})
)

# Mean abs error of last 5 feedback trials per pid-session
last_feedback = feedback_df.groupby(['pid', 'session']).tail(5)
feedback_means = (
    last_feedback.groupby(['pid', 'session', 'condition'])['abs_error']
    .mean()
    .reset_index()
    .rename(columns={'abs_error': 'feedback_abs_error'})
)

# Compute delta abs error = feedback - baseline
merged = feedback_means.merge(baseline_means, on=['pid', 'session'], how='left')
merged['abs_error_difference'] = merged['feedback_abs_error'] - merged['baseline_abs_error']

# Add BDI
bdi = df_trial[['pid', 'bdi']].drop_duplicates(subset='pid')
merged = merged.merge(bdi, on='pid', how='left')

# Recode condition
merged['condition'] = pd.Categorical(merged['condition'], categories=['neut', 'pos', 'neg'])

# Regress delta error on BDI separately for each condition
ols_results = {}
for cond in ['neut', 'pos', 'neg']:
    df_sub = merged[merged['condition'] == cond]

    print(f"\n========= OLS for {cond} (Feedback − Baseline Abs Error) ============\n")
    model = smf.ols("abs_error_difference ~ bdi", df_sub)
    result = model.fit()
    print(result.summary())

    ols_results[cond] = (df_sub, result)

# Plotting
colors = {'neut': 'gray', 'pos': 'tab:green', 'neg': 'tab:red'}
fig, ax = plt.subplots(figsize=(4, 4))

for cond, (df_sub, result) in ols_results.items():
    x = df_sub['bdi']
    y = df_sub['abs_error_difference']
    color = colors[cond]
    label = cond.title()

    ax.scatter(x, y, alpha=0.6, label=label, color=color, s=40)

    # Regression line
    x_vals = pd.Series(sorted(x))
    y_pred = result.predict(exog=dict(bdi=x_vals))
    ax.plot(x_vals, y_pred, color=color, linewidth=2)

    # Add "n.s." if p > 0.05
    p_val = result.pvalues.get("bdi", np.nan)
    if pd.notnull(p_val) and p_val > 0.05:
        x_annot = x_vals.max()
        y_annot = y_pred.iloc[-1]
        ax.text(x_annot, y_annot, "n.s.",
                color=color, fontsize=10,
                verticalalignment='bottom', horizontalalignment='left')

# Format axes and labels
ax.set_xlabel("BDI Score", fontsize=12)
ax.set_ylabel("ΔAbsolute Error\n(Feedback − Baseline)", fontsize=12)
ax.tick_params(axis='both', labelsize=10)
ax.legend(title="Condition", title_fontsize=11, fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

#%%

"""
Does metacognitive bias predict the change in confidence (feedback-baseline)?
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Load and preprocess
df_a = load_df(EXP=1)
df = df_a.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Compute absolute error
df['abs_error'] = abs(df['estimate'] - df['correct'])

# Keep trial-level data
df_trial = df.drop_duplicates(subset=['pid', 'trial', 'session'], keep='first')
df_trial = df_trial.sort_values(by=['pid', 'session', 'trial'])

# Filter participants with all 3 feedback conditions
feedback_df = df_trial[df_trial['condition'] != 'baseline']
session_counts = feedback_df.groupby('pid')['condition'].nunique()
valid_pids = session_counts[session_counts == 3].index
df_trial = df_trial[df_trial['pid'].isin(valid_pids)]

# Split baseline and feedback
baseline_df = df_trial[df_trial['condition'] == 'baseline']
feedback_df = df_trial[df_trial['condition'] != 'baseline']

# Mean confidence during baseline per pid-session
baseline_means = (
    baseline_df.groupby(['pid', 'session'])['confidence']
    .mean()
    .reset_index()
    .rename(columns={'confidence': 'baseline_confidence'})
)

# Mean confidence of last 5 feedback trials
last_feedback = feedback_df.groupby(['pid', 'session']).tail(5)
feedback_means = (
    last_feedback.groupby(['pid', 'session', 'condition'])['confidence']
    .mean()
    .reset_index()
    .rename(columns={'confidence': 'feedback_confidence'})
)

# Merge and compute ΔConfidence
merged = feedback_means.merge(baseline_means, on=['pid', 'session'], how='left')
merged['delta_confidence'] = merged['feedback_confidence'] - merged['baseline_confidence']

# Average ΔConfidence across sessions per condition
delta_confidence_summary = (
    merged.groupby(['pid', 'condition'])['delta_confidence']
    .mean()
    .reset_index()
)

# Fit LMM to get random intercept per participant (metacognitive bias)
lmm_model = smf.mixedlm("confidence ~ abs_error", feedback_df, groups=feedback_df["pid"])
lmm_result = lmm_model.fit()
print(lmm_result.summary())

# Extract full participant-specific intercepts (metacognitive bias)
random_effects = lmm_result.random_effects
intercepts_df = pd.DataFrame({
    "pid": list(random_effects.keys()),
    "metacognitive_bias": [
        lmm_result.fe_params["Intercept"] + v["Group"] for v in random_effects.values()
    ]
})

# Merge with delta confidence
df_plot = delta_confidence_summary.merge(intercepts_df, on='pid', how='left')

# Regress delta confidence on metacognitive bias per condition
ols_results = {}
for cond in ['neut', 'pos', 'neg']:
    df_sub = df_plot[df_plot['condition'] == cond]
    model = smf.ols("delta_confidence ~ metacognitive_bias", data=df_sub)
    result = model.fit()
    ols_results[cond] = (df_sub, result)

    print(f"\n========== OLS for {cond.upper()} ==========")
    print(result.summary())

# Plotting
colors = {'neut': 'gray', 'pos': 'green', 'neg': 'red'}
fig, ax = plt.subplots(figsize=(4, 4))

for cond, (df_sub, result) in ols_results.items():
    x = df_sub['metacognitive_bias']
    y = df_sub['delta_confidence']
    color = colors[cond]
    label = cond.title()

    ax.scatter(x, y, alpha=0.6, label=label, color=color, s=40)

    x_vals = pd.Series(sorted(x))
    y_pred = result.predict(exog=dict(metacognitive_bias=x_vals))
    ax.plot(x_vals, y_pred, color=color, linewidth=2)

    # Annotate non-significance
    p_val = result.pvalues.get("metacognitive_bias", np.nan)
    if pd.notnull(p_val) and p_val > 0.05:
        ax.text(x_vals.max(), y_pred.iloc[-1], "n.s.",
                color=color, fontsize=14,
                verticalalignment='bottom', horizontalalignment='left')
    else:
        ax.text(x_vals.max()+1, y_pred.iloc[-1]-4, "*",
                color='k', fontsize=14,
                verticalalignment='bottom', horizontalalignment='left')

# Aesthetics
ax.set_xlabel("Metacognitive Bias", fontsize=14)
ax.set_ylabel("ΔConfidence\n(Feedback − Baseline)", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.legend(title_fontsize=13, fontsize=12)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()

save_folder = r"C:/Users/carll/OneDrive/Skrivbord/Oxford/DPhil/metacognition-learning/comparative_models/results/Fixed_feedback/analysis"
file_name = 'Confidence_change_vs_metacognition'
save_path = os.path.join(save_folder, file_name)
fig.savefig(f'{save_path}.png', dpi=300)


plt.show()



