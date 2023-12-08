#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress
from IPython.core.display import HTML
import sklearn
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.anova import anova_lm, AnovaRM

def width(axes):
    """Format the axes of a matplotlib figure."""
    line_width = 3
    axes.spines['left'].set_linewidth(line_width)
    axes.spines['bottom'].set_linewidth(line_width)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.tick_params('both', length=12, width=line_width)

def correct_condition(dataframe): #update_baseline_condition
    """Update the condition to 'baseline' based on trial and block criteria."""
    conditions = [(16, 1), (11, 2), (11, 3)]
    for trial, block in conditions:
        dataframe.loc[(dataframe['trial'] == trial) & (dataframe['block'] == block), 'condition'] = 'baseline'
    return dataframe

def standardise(dataframe, variable): #standardize_variable
    """Standardize a variable in the dataframe."""
    dataframe[f'{variable}_c'] = (
        dataframe[variable] - dataframe[variable].mean()) / dataframe[variable].std()
    return dataframe
