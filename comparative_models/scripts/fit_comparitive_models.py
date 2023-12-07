# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:02:33 2023

@author: carll
"""

# Analysing data 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
from src.utility_functions import add_session_column
from src.models import (
                            fit_random_model,
                            fit_random_model_w_bias,
                            fit_win_stay_lose_shift,
                            fit_rw_symmetric_LR
                            
                            )


# Import data 
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
fixed_feedback_data_path = r'fixed_feedback/data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)


# Statistics for each participant, over all conditions
nll_vector_random_p = []
AIC_vector_random_p = []
BIC_vector_random_p = []

nll_vector_bias_p = []
AIC_vector_bias_p = []
BIC_vector_bias_p = []

nll_vector_win_stay_p  = []
AIC_vector_win_stay_p  = []
BIC_vector_win_stay_p  = []


alpha_vector_rw_symm_p = []
sigma_vector_rw_symm_p = []
win_b_vector_rw_symm_p = []
nll_vector_rw_symm_p  = []
AIC_vector_rw_symm_p  = []
BIC_vector_rw_symm_p  = []


# Loop over participants
for participant in tqdm(df.pid.unique(), total=len(df.pid.unique())):
    
    # Current participant data only
    df_p = df[df['pid'] == participant]

    # Get session number 
    df_p = add_session_column(df_p)
    

    nll_vector_random = []
    AIC_vector_random = []
    BIC_vector_random = []
    
    mean_vector_bias = []
    sigma_vector_bias = []
    win_b_vector_bias = []
    nll_vector_bias = []
    AIC_vector_bias = []
    BIC_vector_bias = []
    
    mean_vector_win_stay = []
    sigma_vector_win_stay = []
    win_b_vector_win_stay = []
    nll_vector_win_stay  = []
    AIC_vector_win_stay  = []
    BIC_vector_win_stay  = []
    
    
    alpha_vector_rw_symm = []
    sigma_vector_rw_symm = []
    win_b_vector_rw_symm = []
    nll_vector_rw_symm  = []
    AIC_vector_rw_symm  = []
    BIC_vector_rw_symm  = []
    
    # Loop over sessions
    for session in df_p.session.unique():
        
        # Get current session data, one row per trial
        df_s = df_p[df_p.session == session]
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        # Only feedback trials 
        df_s = df_s[df_s.condition != 'baseline']
        
        
        # Get variables 
        confidence = df_s.estimate.values 
        feedback = df_s.feedback.values
        n_trials=len(df_s)
        
    
        # Random confidence model
        nll_random_model = fit_random_model(prediction_range=100,
                                            n_trials=n_trials)
        k = 0
        random_model_aic = 2*k + 2*nll_random_model
        random_model_bic = k*np.log(n_trials) + 2*nll_random_model
        
        nll_vector_random.append(nll_random_model)
        AIC_vector_random.append(random_model_aic)
        BIC_vector_random.append(random_model_bic)
        
        
        # Biased confidence model
        results = fit_random_model_w_bias(confidence=confidence, 
                                                      n_trials=n_trials)
        
        best_mean = results[0] 
        best_std = results[1] 
        nll = results[2] 
        aic = results[3] 
        bic = results[4] 
        pseudo_r2 = results[5]
        nll_vector_bias.append(nll)
        AIC_vector_bias.append(aic)
        BIC_vector_bias.append(bic)
         
        
        # Win-stay-lose-shift-model
        results_win_stay = fit_win_stay_lose_shift(confidence,
                                                   feedback,
                                                   n_trials)
        
        best_mean = results_win_stay[0] 
        best_std = results_win_stay[1] 
        best_win_boundary = results_win_stay[2]
        nll_vector_win_stay.append(results_win_stay[3]) 
        AIC_vector_win_stay.append(results_win_stay[4])
        BIC_vector_win_stay.append(results_win_stay[5])
        
       
        
        # Rescorla wagner model 
        results_rw_symm = fit_rw_symmetric_LR(confidence, feedback, n_trials)
        
        best_alpha = results_rw_symm[0] 
        best_std = results_rw_symm[1] 
        nll = results_rw_symm[2]
        aic = results_rw_symm[3]
        bic = results_rw_symm[4] 
        pseudo_r2 = results_rw_symm[5] 
        
        alpha_vector_rw_symm.append(best_alpha)
        sigma_vector_rw_symm.append(best_std)
        nll_vector_rw_symm.append(nll)
        AIC_vector_rw_symm.append(aic)
        BIC_vector_rw_symm.append(bic)
        
        
    nll_vector_random_p.append(np.mean(nll_vector_random))
    AIC_vector_random_p.append(np.mean(AIC_vector_random))
    BIC_vector_random_p.append(np.mean(BIC_vector_random))
    
    nll_vector_bias_p.append(np.mean(nll_vector_bias))
    AIC_vector_bias_p.append(np.mean(AIC_vector_bias))
    BIC_vector_bias_p.append(np.mean(BIC_vector_bias))
    
    nll_vector_win_stay_p.append(np.mean(nll_vector_win_stay))
    AIC_vector_win_stay_p.append(np.mean(AIC_vector_win_stay))
    BIC_vector_win_stay_p.append(np.mean(BIC_vector_win_stay))
    
    alpha_vector_rw_symm_p.append(np.mean(alpha_vector_rw_symm))
    sigma_vector_rw_symm_p.append(np.mean(sigma_vector_rw_symm))
    nll_vector_rw_symm_p.append(np.mean(nll_vector_rw_symm))
    AIC_vector_rw_symm_p.append(np.mean(AIC_vector_rw_symm))
    BIC_vector_rw_symm_p.append(np.mean(BIC_vector_rw_symm))
    



#%% Get mean and sem 

# Data for random model
random_model_mean_nll = np.mean(nll_vector_random_p)
random_model_sem_nll = np.std(nll_vector_random_p) / np.sqrt(len(nll_vector_random_p))
random_model_mean_aic = np.mean(AIC_vector_random_p)
random_model_sem_aic = np.std(AIC_vector_random_p) / np.sqrt(len(AIC_vector_random_p))
random_model_mean_bic = np.mean(BIC_vector_random_p)
random_model_sem_bic = np.std(BIC_vector_random_p) / np.sqrt(len(BIC_vector_random_p))

# Data for bias model
bias_model_mean_nll = np.mean(nll_vector_bias_p)
bias_model_sem_nll = np.std(nll_vector_bias_p) / np.sqrt(len(nll_vector_bias_p))
bias_model_mean_aic = np.mean(AIC_vector_bias_p)
bias_model_sem_aic = np.std(AIC_vector_bias_p) / np.sqrt(len(AIC_vector_bias_p))
bias_model_mean_bic = np.mean(BIC_vector_bias_p)
bias_model_sem_bic = np.std(BIC_vector_bias_p) / np.sqrt(len(BIC_vector_bias_p))

# Data for win-stay model
win_stay_model_mean_nll = np.mean(nll_vector_win_stay_p)
win_stay_model_sem_nll = np.std(nll_vector_win_stay_p) / np.sqrt(len(nll_vector_win_stay_p))
win_stay_model_mean_aic = np.mean(AIC_vector_win_stay_p)
win_stay_model_sem_aic = np.std(AIC_vector_win_stay_p) / np.sqrt(len(AIC_vector_win_stay_p))
win_stay_model_mean_bic = np.mean(BIC_vector_win_stay_p)
win_stay_model_sem_bic = np.std(BIC_vector_win_stay_p) / np.sqrt(len(BIC_vector_win_stay_p))


# RW symetric LR model 
alpha_vector_rw_symm_p

rw_symm_model_mean_nll = np.mean(nll_vector_rw_symm_p)
rw_symm_model_sem_nll = np.std(nll_vector_rw_symm_p) / np.sqrt(len(nll_vector_rw_symm_p))
rw_symm_model_mean_aic = np.mean(AIC_vector_rw_symm_p)
rw_symm_model_sem_aic = np.std(AIC_vector_rw_symm_p) / np.sqrt(len(AIC_vector_rw_symm_p))
rw_symm_model_mean_bic = np.mean(BIC_vector_rw_symm_p)
rw_symm_model_sem_bic = np.std(BIC_vector_rw_symm_p) / np.sqrt(len(BIC_vector_rw_symm_p))

#%% Plot mean across conditions


# Set up the plot
fig, ax = plt.subplots(figsize=(6, 4))

# Define x-coordinates for each metric group and offsets for each model within the group
x = np.arange(3)  # Base x-coordinates for each metric
offset = 0.2  # Offset for each model within a metric group

# Colors for each model (added one more color for RW symetric LR model)
colors = ['blue', 'green', 'red', 'purple']  # Random, Bias, Win-Stay, RW symmetric LR

# Plotting NLL for each model
model_means = [random_model_mean_nll, bias_model_mean_nll, win_stay_model_mean_nll, rw_symm_model_mean_nll]
model_sems = [random_model_sem_nll, bias_model_sem_nll, win_stay_model_sem_nll, rw_symm_model_sem_nll]
model_names = ["Random", "Biased", "Win-Stay-Lose-Shift", "RW Symmetric LR"]


for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(x[0] + offset * (i - 1.5), mean, yerr=sem, fmt='o', capsize=10, color=colors[i], label=model_names[i])

# Plotting AIC for each model
model_means = [random_model_mean_aic, bias_model_mean_aic, win_stay_model_mean_aic, rw_symm_model_mean_aic]
model_sems = [random_model_sem_aic, bias_model_sem_aic, win_stay_model_sem_aic, rw_symm_model_sem_aic]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(x[1] + offset * (i - 1.5), mean, yerr=sem, fmt='o', capsize=10, color=colors[i])

# Plotting BIC for each model
model_means = [random_model_mean_bic, bias_model_mean_bic, win_stay_model_mean_bic, rw_symm_model_mean_bic]
model_sems = [random_model_sem_bic, bias_model_sem_bic, win_stay_model_sem_bic, rw_symm_model_sem_bic]

for i, (mean, sem) in enumerate(zip(model_means, model_sems)):
    ax.errorbar(x[2] + offset * (i - 1.5), mean, yerr=sem, fmt='o', capsize=10, color=colors[i])

# Customizing the Axes
ax.set_xticks(x)
ax.set_xticklabels(['NLL', 'AIC', 'BIC'])
ax.set_ylabel('Model fits')

# Remove top and right spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adding a legend
ax.legend(model_names)

plt.tight_layout()
plt.show()
