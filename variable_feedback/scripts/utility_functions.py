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
import pingouin as pg


def make_figure(axes):
    width = 3
    axes.spines['left'].set_linewidth(width)
    axes.spines['bottom'].set_linewidth(width)
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.tick_params('both', length=12, width=width)


def correct_condition(df):
    df.loc[(df.trial==16) & (df.block==1), 'condition']='baseline'
    df.loc[(df.trial==11) & (df.block==2), 'condition']='baseline'
    df.loc[(df.trial==11) & (df.block==3), 'condition']='baseline'
    return df

def standardise(df, var):
    df[f'{var}_c']=(df[f'{var}']-df[f'{var}'].mean())/df[f'{var}'].std()
    return df
