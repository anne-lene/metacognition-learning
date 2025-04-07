#%% -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 15:10:46 2025

@author: carll
"""
import sys
import os

# Add the project root to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils import (load_df, load_sim_df)
from matplotlib import pyplot as plt 
import pandas 
import numpy

plt.ion() 
df = load_df(EXP=1)
print(df.head())

df_sim = load_sim_df(EXP=2)
print(df_sim.head())

fig, ax = plt.subplots(1,1, figsize=(6,4))

ax.plot([1,2,3], [1,2,3])
plt.show()


# %%
