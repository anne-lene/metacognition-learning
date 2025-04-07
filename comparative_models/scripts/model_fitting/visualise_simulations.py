# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:22:56 2025

@author: carll
"""

# Visualise simulations

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression


def plot_model_results(df):
    """
    Plot results for each model with two columns: individual session outcomes
    and mean confidence over time.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['confidence_sim', 'model', 'trial', 'session', 'pid'].
    """
    # Get unique model names to determine the number of rows
    model_names = df['model'].unique()
    n_models = len(model_names)

    # Set up a 2-column subplot structure
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 4 * n_models), sharex='col', sharey='row')
    if n_models == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    # Plot each model as separate rows
    for model_idx, model_name in enumerate(model_names):
        # Separate the two subplots for this model
        ax_individual = axes[model_idx][0]
        ax_mean = axes[model_idx][1]

        # Filter data for the current model
        model_df = df[df['model'] == model_name]

        # Plot each session's confidence for the current model in the first column
        for session_id in model_df['session'].unique():
            session_df = model_df[model_df['session'] == session_id]
            ax_individual.plot(session_df['trial'], session_df['confidence_sim'],
                               label=f'Session {session_id}', marker='o')

        # Calculate and plot the mean confidence over trials for the current model
        mean_confidence = model_df.groupby('trial')['confidence_sim'].mean()
        ax_mean.plot(mean_confidence.index, mean_confidence.values, color='b', marker='o', linestyle='-', linewidth=2)

        # Set titles and labels
        ax_individual.set_title(f'{model_name} - Individual Sessions')
        ax_mean.set_title(f'{model_name} - Mean Confidence Over Trials')
        ax_individual.set_ylabel('Confidence')
        ax_individual.spines[['top', 'right']].set_visible(False)
        ax_mean.spines[['top', 'right']].set_visible(False)

        # Set y-axis limits
        ax_individual.set_ylim(-5, 105)
        ax_mean.set_ylim(-5, 105)

    # Label the shared x-axis
    axes[-1][0].set_xlabel('Trial')
    axes[-1][1].set_xlabel('Trial')
    plt.tight_layout()
    plt.show()

# Load data
sim_data = pd.read_csv('simulated_conf_EXP2_data_more_deterministic.csv')

# Plot
plot_model_results(sim_data)
