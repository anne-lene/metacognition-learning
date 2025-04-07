# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:54:23 2024

@author: carll
"""

# Model Recovery

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('sim_EXP2_model_metrics_sessions_CV.xlsx',
                   index_col='Unnamed: 0')

# Rename 'nll_array_rw_cond' to 'nll_array_rw_cond_p' for consistency
df = df.rename(columns={'nll_array_rw_cond': 'nll_array_rw_cond_p'})

# Create simulation model
model_order = ['bias', 'wsls', 'rw', 'rw_cond', 'ck', 'rwck', 'rwpd']
df['sim_model'] = np.tile(model_order, len(df) // len(model_order) + 1)[:len(df)]

#df['simulated_model'] = np.tile(model_order, len(df) // len(model_order))[:len(df)]

# Define model names and NLL columns
model_names = ['random', 'bias', 'win_stay', 'rw_symm', 'rw_cond', 'ck', 'rwck', 'delta_p_rw']
nll_columns = [f'nll_array_{name}_p' for name in model_names]

# Identify the best fitting model by finding the model with the minimum NLL for each row
df['fitted_model'] = df[nll_columns].idxmin(axis=1).str.replace('nll_array_', '').str.replace('_p', '')

# Create a confusion matrix: rows = simulated model, columns = fitted model
confusion_matrix = pd.crosstab(df['sim_model'], df['fitted_model'], normalize='index')

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix, annot=True, cmap="viridis", fmt=".2f",
            xticklabels=model_order, yticklabels=model_order)
plt.xlabel("Fit Model")
plt.ylabel("Simulated Model")
plt.title("Confusion Matrix: p(Fit Model | Simulated Model)")
plt.show()


