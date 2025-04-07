# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:58:42 2025

@author: carll
"""


# Plotting Distribution of best fit paramters for Comparative Models on Experiment 2

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
#df_m = pd.read_excel('EXP2_model_metrics_sessions_CV_v3_rand_column_change.xlsx')
df_m = pd.read_excel('EXP2_model_metrics_sessions_CV_v10.xlsx')
df_m = df_m.rename(columns=lambda col: col if col.endswith('_p') else f"{col}_p")


params = ['mean_array_bias_p', 'sd_array_bias_p',
          'std_WSLS_array_p','win_boundary_WSLS_array_p',
          'alpha_array_rw_symm_p', 'sigma_array_rw_symm_p',
          'alpha_neut_array_rw_cond_p', 'alpha_pos_array_rw_cond_p', 'alpha_neg_array_rw_cond_p', 'sigma_array_rw_cond_p',
          'alpha_array_ck_p', 'sigma_array_ck_p', 'beta_array_ck_p',
          'alpha_array_rwck_p', 'alpha_ck_array_rwck_p','sigma_array_rwck_p', 'sigma_ck_array_rwck_p', 'beta_array_rwck_p', 'ck_beta_array_rwck_p',
          'alpha_array_delta_p_rw_p', 'sigma_array_delta_p_rw_p', 'w_rw_array_delta_p_rw_p', 'w_delta_p_array_delta_p_rw_p']


for param in params:
    fig, ax = plt.subplots(1,1, figsize=(5,4))
    plt.title(param)
    plt.hist(df_m[param], bins=20)
    plt.show()



