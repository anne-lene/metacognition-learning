# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 01:58:54 2025

@author: carll
"""

import pandas as pd
import os

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Relative path
relative_path = "../../results/variable_feedback/model_comparison/model_and_param_recovery"

# Construct the full path to the CSV file
file_path = os.path.normpath(os.path.join(script_dir, relative_path))
file_name = r"model_fits_EXP2_sim_test"
save_path = os.path.join(relative_path, file_path, file_name)

# Construct and save dataframe
df_m = pd.DataFrame({'data': [1,2,3,4,5,6,7,8,9,10]})
df_m.to_excel(f"{save_path}.xlsx") # save to specified directory