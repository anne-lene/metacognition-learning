# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:29:24 2024

@author: carll
"""


# Regression analysis

import numpy as np
from src.utility_functions import (add_session_column)
import pandas as pd
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import pearsonr
from patsy import dmatrices

# Import data
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_path = r"fixed_feedback"
fixed_feedback_data_path = r'data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, experiment_path,
                         fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

# Add session column
df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

error_baseline_list = []
performance_baseline_list  = []
error_task_list  = []
performance_task_list  = []
bdi_list  = []
condition_list = []
pid_list = []
pid_nr_list = []
confidence_baseline_list = []
confidence_task_list = []
p_avg_task_list = []
p_avg_baseline_list = []
error_change_baseline_list = []
error_change_task_list = []
estimate_baseline_list = []
estimate_task_list = []
subtrial_variance_baseline = []
subtrial_std_baseline = []
subtrial_variance_task = []
subtrial_std_task = []

subtrial_error_mean_baseline_list = []
subtrial_error_variance_baseline_list = []
subtrial_error_std_baseline_list = []
subtrial_error_mean_task_list = []
subtrial_error_variance_task_list = []
subtrial_error_std_task_list = []

feedback_list = []

# Loop over participants
for pid_nr, participant in enumerate(tqdm(df.pid.unique(),
                                     total=len(df.pid.unique()),
                                     desc='Participant loop')):

    df_p = df[df.pid==participant]

    error_baseline_p = []
    performance_baseline_p = []

    error_task_p = []
    performance_task_p = []

    bdi_p = []
    condition_p = []
    pid_p = []
    pid_nr_p = []

    # Loop over sessions
    for session in df_p.session.unique():

        # Get current session data
        df_s = df_p[df_p.session == session].copy()

        # get the mean, std, and var of estimate of across all subtrials
        df_s['mean_estimate'] = df_s.groupby('trial')['estimate'].transform('mean')
        df_s['std_estimate'] = df_s.groupby('trial')['estimate'].transform('std')
        df_s['var_estimate'] = df_s.groupby('trial')['estimate'].transform('var')

        # Calculate the absolute error for each subtrial
        df_s['abs_error'] = (df_s['estimate'] - df_s['correct']).abs()

        # Get the mean, variance and std of absolute error
        df_s['mean_error'] = df_s.groupby('trial')['abs_error'].transform('mean')
        df_s['std_error'] = df_s.groupby('trial')['abs_error'].transform('std')
        df_s['var_error'] = df_s.groupby('trial')['abs_error'].transform('var')

        # Only consider the last subtrial
        df_s = df_s.drop_duplicates(subset='trial', keep='last')

        # Filter out all 'baseline' rows
        df_s_task = df_s[df_s['condition'] != 'baseline']

        # Only take the last 10 trils in baseline
        df_s_baseline = df_s[df_s['condition'] == 'baseline'].tail(10)

        # Get values
        n_trials = len(df_s_task)
        condition = df_s_task.condition.unique()[0]
        confidence = df_s_task.confidence.values
        feedback = df_s_task.feedback.values

        # Depression measure
        bdi = df_s.bdi.values[0]

        # dot estimates
        estimate_baseline = df_s_baseline.mean_estimate.values
        estimate_task = df_s_task.mean_estimate.values

        estimate_std_baseline = df_s_baseline.std_estimate.values
        estimate_std_task = df_s_task.std_estimate.values

        estimate_var_baseline = df_s_baseline.var_estimate.values
        estimate_var_task = df_s_task.var_estimate.values

        # Error stats from all subtrials
        subtrial_error_mean_baseline = df_s_baseline.mean_error.values
        subtrial_error_std_baseline = df_s_baseline.std_error.values
        subtrial_error_var_baseline = df_s_baseline.var_error.values

        subtrial_error_mean_task = df_s_task.mean_error.values
        subtrial_error_std_task = df_s_task.std_error.values
        subtrial_error_var_task = df_s_task.var_error.values

        # Error and performance - brased on last subtrial
        error_baseline = df_s_baseline.estimate.values - df_s_baseline.correct.values
        performance_baseline = abs(error_baseline)

        error_task = df_s_task.estimate.values - df_s_task.correct.values
        performance_task = abs(error_task)

        p_avg_task = df_s_task.pavg.values
        p_avg_baseline = df_s_baseline.pavg.values

        # Confidence
        confidence_task = df_s_task.confidence.values
        confidence_baseline = df_s_baseline.confidence.values

        # Trial-to-Trial change in error
        error_change_baseline = np.diff(performance_baseline)
        error_change_task = np.diff(performance_task)

        # Save to list
        subtrial_variance_baseline.append(estimate_var_baseline)
        subtrial_std_baseline.append(estimate_std_baseline )
        subtrial_variance_task.append(estimate_var_task)
        subtrial_std_task.append(estimate_std_task)

        subtrial_error_mean_baseline_list.append(subtrial_error_mean_baseline)
        subtrial_error_variance_baseline_list.append(subtrial_error_var_baseline)
        subtrial_error_std_baseline_list.append(subtrial_error_std_baseline)
        subtrial_error_mean_task_list.append(subtrial_error_mean_task)
        subtrial_error_variance_task_list.append(subtrial_error_var_task)
        subtrial_error_std_task_list.append(subtrial_error_std_task)

        error_baseline_list.append(error_baseline)
        performance_baseline_list.append(performance_baseline)
        error_task_list.append(error_task)
        performance_task_list.append(performance_task)
        bdi_list.append(bdi)
        condition_list.append(condition)
        pid_list.append(participant)
        pid_nr_list.append(pid_nr)
        p_avg_task_list.append(p_avg_task)
        p_avg_baseline_list.append(p_avg_baseline)
        error_change_baseline_list.append(error_change_baseline)
        error_change_task_list.append(error_change_task)
        confidence_baseline_list.append(confidence_baseline)
        confidence_task_list.append(confidence_task)
        estimate_baseline_list.append(estimate_baseline)
        estimate_task_list.append(estimate_task)
        feedback_list.append(feedback)


df_a = pd.DataFrame({'error_baseline': error_baseline_list,
                     'performance_baseline': performance_baseline_list,
                     'error_task': error_task_list,
                     'performance_task': performance_task_list,
                     'bdi': bdi_list,
                     'condition_list': condition_list,
                     'pid': pid_list,
                     'pid_nr': pid_nr_list,
                     'confidence_baseline': confidence_baseline_list,
                     'confidence_task': confidence_task_list,
                     'p_avg_task': p_avg_task_list,
                     'p_avg_baseline': p_avg_baseline_list,
                     'error_change_baseline':error_change_baseline_list,
                     'error_change_task': error_change_task_list,
                     'estimate_baseline': estimate_baseline_list,
                     'estimate_task': estimate_task_list,
                     'subtrial_std_baseline': subtrial_std_baseline,
                     'subtrial_var_baseline': subtrial_variance_baseline,
                     'subtrial_std_task': subtrial_std_task,
                     'subtrial_var_task': subtrial_variance_task,
                     'subtrial_error_mean_baseline': subtrial_error_mean_baseline_list,
                     'subtrial_error_variance_baseline': subtrial_error_variance_baseline_list,
                     'subtrial_error_std_baseline': subtrial_error_std_baseline_list,
                     'subtrial_error_mean_task': subtrial_error_mean_task_list,
                     'subtrial_error_variance_task': subtrial_error_variance_task_list,
                     'subtrial_error_std_task': subtrial_error_std_task_list,
                     'feedback': feedback_list,

                     })



# Initialize variables
conditions = ['neut', 'pos', 'neg']
colors = {'neut': 'grey', 'pos': 'green', 'neg': 'red'}



#%% Q: Does depression symptoms predict confidence bias and sensitivity?
  # A: Only Bias

biases_baseline_list = []
biases_task_list = []
sensitivity_baseline = []
sensitivity_task = []
condition = []
pid_list = []
bdi_list = []
# Iterate over participants
for index, session in df_a.iterrows():


    # Get sensitivity
    corr_baseline, p_v_baseline = pearsonr(session['confidence_baseline'],
                                           session['subtrial_error_mean_baseline'])

    corr_task, p_v_task = pearsonr(session['confidence_task'],
                                   session['subtrial_error_mean_task'])

    # If constant confidene - remove participant
    if np.isnan(corr_baseline) or np.isnan(corr_task):
        continue

    # Get bias
    baseline_bias = np.mean(session.confidence_baseline)
    task_bias = np.mean(session.confidence_task)

    biases_baseline_list.append(baseline_bias)
    biases_task_list.append(task_bias)



    sensitivity_baseline.append(corr_baseline)
    sensitivity_task.append(corr_task)

    # Get condition
    condition.append(session.condition_list)

    # Get pid
    pid = session.pid
    pid_list.append(pid)

    # Get bdi
    bdi_series = df_a[df_a['pid'] == pid].bdi
    bdi = bdi_series.dropna().values[0]
    bdi_list.append(bdi)



fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 5))

# Scatter plots
ax0.scatter(bdi_list, biases_baseline_list, c='C0')
ax1.scatter(bdi_list, biases_task_list, c='C0')
ax2.scatter(bdi_list, sensitivity_baseline, c='C1')
ax3.scatter(bdi_list, sensitivity_task, c='C1')

# Lists to store axes, x data, and y data
axes = [ax0, ax1, ax2, ax3]
x_data = [bdi_list, bdi_list,
          bdi_list, bdi_list]
y_data = [biases_baseline_list, biases_task_list,
          sensitivity_baseline, sensitivity_task]

# Calculate and annotate each plot with Pearson's r and p-value
for ax, x, y in zip(axes, x_data, y_data):

    # Pearson correlation
    r_value, p_value = pearsonr(x, y)

    # Add trend line
    z = np.polyfit(x, y, 1)
    p_poly = np.poly1d(z)
    ax.plot(x, p_poly(x), "r--")

    # Annotate with r and p-value
    ax.annotate(f'r = {r_value:.2f}\np = {p_value:.5f}', xy=(0.75, 1),
                xycoords='axes fraction', fontsize=10, verticalalignment='top')

    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%% Q: Does condition effect correlation?
#   A: no.

df_2 = pd.DataFrame({'bdi_list': bdi_list,
                   'biases_baseline_list': biases_baseline_list,
                   'biases_task_list': biases_task_list,
                   'sensitivity_baseline': sensitivity_baseline,
                   'sensitivity_task': sensitivity_task,
                   'condition': condition,
                   'pid_list': pid_list,
                   })

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(8, 5))

# Define plot axes and corresponding columns
plot_axes = [(ax0, 'bdi_list', 'biases_baseline_list'),
             (ax1, 'bdi_list', 'biases_task_list'),
             (ax2, 'bdi_list', 'sensitivity_baseline'),
             (ax3, 'bdi_list', 'sensitivity_task')]

colors = ['grey', 'green', 'red']  # Assuming 3 conditions, assign different colors
condition_list = ['neut', 'pos', 'neg']

for ax, x_col, y_col in plot_axes:

    count = 0
    for cond, color in zip(condition_list, colors):

        # Filter data based on condition
        filtered_df = df_2[df_2['condition'] == cond]


        # Extract x and y values
        x = filtered_df[x_col].values
        y = filtered_df[y_col].values

        # Check if x and y are not empty
        if len(x) > 0 and len(y) > 0:
            # Pearson correlation
            r_value, p_value = pearsonr(x, y)

            # Add trend line
            z = np.polyfit(x, y, 1)
            p_poly = np.poly1d(z)
            ax.plot(x, p_poly(x), "--", color=color, alpha=1, ls='--')
            ax.scatter(x, y, c=color, alpha=0.5, lw=0.1)

            # Annotate with r and p-value
            ax.annotate(f'Cond {cond}: r={r_value:.2f}, p={p_value:.5f}',
                        xy=(0.05, 1.1 + count),
                        xycoords='axes fraction', fontsize=10,
                        verticalalignment='top', color=color)

        print(cond)
        count += 0.1

    ax.spines[['top', 'right']].set_visible(False)
    #ax.legend()


plt.tight_layout()
plt.show()


#%% Does trial-by-trial change in feedback predict change in confidence?

import pandas as pd
import statsmodels.api as sm
import numpy as np

# Assuming df is your DataFrame

# Initialize an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Participant', 'Session', 'Slope',
                                   'Intercept', 'R-Squared', 'P-Value'])

slopes = []
conditions = []
intercepts = []
last_y_preds = []
fig, (ax, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(6,8))
for (pid, session), group in df.groupby(['pid', 'session']):


    # Calculate the difference between 'estimate' and 'correct' for each entry
    group['error'] = group['estimate'] - group['correct']

    # Get current abs error
    current_error = group.groupby(['subtrial'])['error'].mean()
    error_change = current_error - abs(group['estimate'].shift(1) -
                                       group['correct'].shift(1))

    # get bdi
    bdi = df[df['pid'] == pid]['bdi'].dropna().unique()[0]

    # Only have 1 value/row per trial
    group = group[group['subtrial']==3]

    # Ensure the group is sorted by trial number if not already
    group = group.sort_values('trial')

    # Keep only the last 20 trials
    group = group.tail(20)
    #group = group.head(10)

    # Shift the feedback to align with the next trial's confidence
    previous_feedback = group['feedback'].shift(1)
    current_confidence = group['confidence']
    confident_update = group['confidence'] - group['confidence'].shift(1)
    feedback_update = group['feedback'].shift(1) - group['feedback'].shift(2)

    # Get current abs error
    current_error = abs(group['estimate']-group['correct'])
    error_change = current_error - abs(group['estimate'].shift(1) -
                                       group['correct'].shift(1))

    # Get current relative error
    current_rel_error = group.pavg
    rel_error_change = current_rel_error - group.pavg.shift(1)

    # Get interaction
    error_and_feedback = error_change+feedback_update


    # Drop the first row to align previous feedback with current confidence
    previous_feedback = previous_feedback[2:].reset_index(drop=True)
    current_confidence = current_confidence[2:].reset_index(drop=True)
    current_error = current_error[2:].reset_index(drop=True)
    confident_update = confident_update[2:].reset_index(drop=True)
    feedback_update = feedback_update[2:].reset_index(drop=True)
    error_change = error_change[2:].reset_index(drop=True)
    error_and_feedback = error_and_feedback[2:].reset_index(drop=True)
    rel_error_change = rel_error_change[2:].reset_index(drop=True)


    # Check if data is sufficient for regression
    if len(previous_feedback) > 1 and len(current_confidence) > 1:

        data_df = pd.DataFrame({'confident_update': confident_update,
                                'feedback_update': feedback_update,
                                'error_change': error_change,
                                'rel_error_change': rel_error_change,

                                })

        # Prepare data for regression
        Y, X = dmatrices('confident_update ~ feedback_update',
                         data=data_df,
                         return_type='dataframe')

        # Perform linear regression
        model = sm.OLS(Y, X)
        results = model.fit()
        Y_pred = results.predict(X)

        if len(results.params) > 1:
            # Store results if there are enough parameters (intercept and slope)
            results_df = pd.concat([results_df, pd.DataFrame([{
                'Participant': pid,
                'Session': session,
                'Slope1': results.params[1],
            #    'Slope2': results.params[2],
                'Intercept': results.params[0],
                'R-Squared': results.rsquared,
                'P-Value': results.f_pvalue,
                'P-Value1': results.pvalues[1] ,
            #    'P-Value2': results.pvalues[2],
#                'P-Value3': results.pvalues[3],
                'params': results.params.index,
                'condition': group.condition.unique()[0],
                'bdi': bdi,
                'confident_update': confident_update,
                'feedback_update': feedback_update,
                'error_change': error_change,
                'rel_error_change': rel_error_change,




            }])], ignore_index=True)



            data_df = pd.DataFrame({'confidence': group['confidence'],
                                    'trial': range(len(group['confidence']))
                                    })

            # Prepare data for regression
            Y, X = dmatrices('confidence ~ trial',
                             data=data_df,
                             return_type='dataframe')

            # Perform linear regression
            model = sm.OLS(Y, X)
            results = model.fit()
            Y_pred = results.predict(X)

            d = {'neut': 'grey', 'neg': 'red', 'pos':'green'}
            ax.plot(range(len(X)), Y_pred,
                       lw=0.1, c=d[group.condition.unique()[0]])
            slopes.append(results.params[1])
            intercepts.append(results.params[0])
            conditions.append(group.condition.unique()[0])
            last_y_preds.append(Y_pred.values[-1])

df_hist = pd.DataFrame({'slopes': slopes,
                        'conditions': conditions,
                        'intercepts': intercepts,
                        'last_y_preds': last_y_preds,
                        })

d = {'neut': 'grey', 'neg': 'red', 'pos':'green'}
for (cond), group in df_hist.groupby(['conditions']):

    ax2.hist(group.slopes, bins=30, color=d[cond[0]], alpha=0.4)
    ax2.axvline(np.mean(group.slopes), color=d[cond[0]], ls='-')
    ax3.hist(group.intercepts, bins=30, color=d[cond[0]], alpha=0.4)
    ax3.axvline(np.mean(group.intercepts), color=d[cond[0]], ls='-')

    ax4.hist(group.last_y_preds, bins=30, color=d[cond[0]], alpha=0.4)
    ax4.axvline(np.mean(group.last_y_preds), color=d[cond[0]], ls='-')


ax.set_xlabel('Trials')
ax2.set_xlabel('Slope')
ax3.set_xlabel('Intercept')
ax4.set_xlabel('Last trial')

ax.set_ylabel('confidence')
ax2.set_ylabel('count')
ax3.set_ylabel('count')
ax4.set_ylabel('count')

ax.spines[['top', 'right']].set_visible(False)
ax2.spines[['top', 'right']].set_visible(False)
ax3.spines[['top', 'right']].set_visible(False)
ax4.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()
# Display the results
print(results_df)


#%% Histogram

fig, (ax) = plt.subplots(1,1, figsize=(8,4))

ax.set_title(results_df.params[0][1])
ax.hist(results_df['R-Squared'], bins=20)
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylabel('count')
ax.set_xlabel('r squared')
ax.set_xlim(0,1)
ax.set_xlim(0,1)
plt.tight_layout()
plt.show()



#%%

fig, (ax, ax2, ax3) = plt.subplots(3,1, figsize=(8,4))
for cond in ['neg', 'neut', 'pos']:
    df_plot = results_df[results_df['condition']==cond]


    ax.set_title(df_plot.params.index[1])
    ax.hist(df_plot['P-Value1'], bins=20)
    ax2.set_title(df_plot.params.index[2])
    ax2.hist(df_plot['P-Value2'], bins=20)
    #ax3.set_title(results.params.index[3])
    #ax3.hist(results_df['P-Value3'], bins=200)
    ax.axvline(0.05, ls='--', c='k')
    plt.tight_layout()

    ax.spines[['top', 'right']].set_visible(False)
    ax2.spines[['top', 'right']].set_visible(False)
    ax3.spines[['top', 'right']].set_visible(False)
    plt.show()

#%% Scatter plot - is the correlation between

fig, (ax) = plt.subplots(1,1, figsize=(6,4))

for cond, color in zip(['neg', 'neut', 'pos'], ['red', 'grey', 'green']):

    df_plot = results_df[results_df['condition']==cond]

    ax.set_title(df_plot.params.index[1])
    ax.scatter(df_plot['P-Value1'], df_plot['P-Value2'], color='blue')




ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.show()

#%%

results_df['P-Value1']


fig, (ax) = plt.subplots(1,1, figsize=(6,4))
ax.set_title(results.params.index[1])
ax.scatter(results_df['P-Value1'], results_df['bdi'],)



plt.tight_layout()

ax.spines[['top', 'right']].set_visible(False)
ax2.spines[['top', 'right']].set_visible(False)
ax3.spines[['top', 'right']].set_visible(False)


#%%
from sklearn.preprocessing import MinMaxScaler
# Re-creating the DataFrame
data = {
    "confident_update": confident_update,
    "feedback_update": feedback_update,
    "error_change": error_change
}

df_example = pd.DataFrame(data)

# Normalizing the data between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized = pd.DataFrame(scaler.fit_transform(df_example), columns=df_example.columns)

# Plotting the normalized data
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(df_normalized['confident_update'], label='confident_update')
ax.plot(df_normalized['feedback_update'], label='feedback_update')
ax.plot(df_normalized['error_change'], label='error_change')
ax.spines[['top', 'right']].set_visible(False)
plt.legend()
plt.show()







