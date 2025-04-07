# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 19:52:52 2025

@author: carll
"""

# Simulate RW model - visualise different random sigma values.
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from src.utils import (add_session_column, load_df)
from src.models import (RWPD_sim,
                        RWP_sim,
                        RWCK_sim,
                        CK_sim,
                        RW_cond_sim,
                        RW_sim,
                        WSLS_sim,
                        bias_model_sim,
                        )

# Import data - Variable feedback condition (Experiment 2)
df = load_df(EXP=2)

# Add session column
df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Get unique pid and session pairs
unique_pairs = df[['pid', 'session']].drop_duplicates()
unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

# Create a list of DataFrames, one for each unique pid-session pair
df_list = [df[(df['pid'] == pid) & (df['session'] == session)].copy()
           for pid, session in unique_pairs_list]

df_s = df_list[0] # Select one session to use as test.

# Participant and session
participant = df_s.pid.unique()[0]
session = df_s.session.unique()[0]

# Remove baseline
df_s = df_s[df_s.condition != 'baseline']

# Calculate absolute trial error as the average across subtrials
df_s['difference'] = abs(df_s['estimate'] - df_s['correct'])
abs_error_avg = df_s.groupby('trial')['difference'].mean()


# Only keep first row of every subtrial (one row = one trial)
df_s = df_s.drop_duplicates(subset='trial', keep='first')

# Ensure performance aligns with filtered df_s
# performance is inverse of abs error
df_s['performance'] = -abs_error_avg.values # df_s['trial'].map(lambda t: -abs_error_avg.loc[t])

# Condition
conditions = df_s.condition.unique()

# N trials
n_trials = len(df_s)

# Calculate trial-by-trial metrics
confidence = df_s.confidence.values
feedback = df_s.feedback.values
performance = -abs_error_avg.values

# Plot
trial = 0
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for _ in range(3):
    # RW model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    rw_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }
    sigma_val = rw_params['sigma']
    print('sigma:', round(sigma_val))

    rw_sim_conf, y_val_rw = RW_sim((rw_params['alpha'], sigma_val), confidence, feedback, n_trials)

    # Extract y values for a single trial (a vector)
    y_values = y_val_rw[trial]

    # Plot the full line
    line, = ax.plot(y_values)

    # Annotate the last point
    x_annotate = len(y_values) - 40
    y_annotate = y_values[-50]
    ax.annotate(f'σ = {sigma_val:.2f}',
                xy=(x_annotate, y_annotate),
                xytext=(x_annotate + 1, y_annotate + 0.01),
                #arrowprops=dict(arrowstyle='->', lw=1),
                fontsize=8,
                color=f'C{_}')

    ax.set_ylim(0, 0.4)

ax.set_xlabel('Confidence')
ax.set_ylabel('Probability')
ax.spines[['top', 'right']].set_visible(False)
plt.show()
