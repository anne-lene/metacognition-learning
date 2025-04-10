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
df_a = load_df(EXP=2)

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

