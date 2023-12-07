import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from statsmodels.graphics.factorplots import interaction_plot
from statsmodels.stats.anova import anova_lm, AnovaRM
import pingouin as pg

def make_figure(ax):
    """
    Styles the axes of a matplotlib figure.

    :param ax: Matplotlib Axes object to be styled.
    """
    width = 3
    ax.spines['left'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params('both', length=12, width=width)

def correct_condition(df):
    """
    Corrects the 'condition' column in the DataFrame based on trial and block values.

    :param df: Pandas DataFrame with 'trial', 'block', and 'condition' columns.
    :return: Pandas DataFrame with corrected 'condition' column.
    """
    condition_corrections = {
        (16, 1): 'baseline',
        (11, 2): 'baseline',
        (11, 3): 'baseline'
    }

    for (trial, block), condition in condition_corrections.items():
        df.loc[(df['trial'] == trial) & (df['block'] == block), 'condition'] = condition

    return df

def standardise(df, var):
    """
    Standardizes variable in the DataFrame.

    :param df: Pandas DataFrame containing the variable to be standardized.
    :param var: Name (string) of the variable to be standardized.
    :return: Pandas DataFrame with the standardized variable.
    """
    column_name = f'{var}_c'
    df[column_name] = (df[var] - df[var].mean()) / df[var].std()
    return df
