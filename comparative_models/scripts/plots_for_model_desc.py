# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 21:40:17 2023

@author: carll
"""

# plots for model text

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy import diff
from tqdm import tqdm
import scipy
import scipy.special
from scipy.special import logsumexp
import math



def create_probability_distribution(num_options=100, mean_option=50, sigma=10):
    """
    Create a normal distribution of probabilities for choosing among 
    a set number of options. The distribution is centered around a specified 
    option (mean) with a given standard deviation (sigma).

    :param num_options: Total number of options (default 100).
    :param mean_option: The option number around which the distribution 
                        is centered (default 50).
    :param sigma: Standard deviation controlling the spread of the 
                  distribution (default 10).
    :return: An array of probabilities representing the normal distribution 
             over the options.
    """
    # Generating option indices
    options = np.arange(num_options)
    
    # Creating a normal distribution around the mean_option with normalization factor
    factor = 1 / (sigma * np.sqrt(2 * np.pi))
    probabilities = factor * np.exp(-0.5 * ((options - mean_option) / sigma) ** 2)

    # Normalize the distribution so that the sum of probabilities equals 1
    probabilities /= probabilities.sum()

    return probabilities

def invert_normal_distribution(probabilities):
    max_prob = np.max(probabilities)  # Find the maximum probability
    inverted = max_prob - probabilities  # Invert the probabilities
    normalized_inverted = inverted / np.sum(inverted)  # Normalize the inverted probabilities
    return normalized_inverted

fig, [ax, ax2] = plt.subplots(1,2, figsize=(8,3), sharex=True, sharey=True)
for x in [20, 50, 80]:
    
    sigma=10
    prob_distribution = create_probability_distribution(num_options=100, 
                                                        mean_option=x, 
                                                        sigma=10)
    
    
    print(sum(prob_distribution))
    ax.scatter(range(100), prob_distribution, label=f"x={x}, std={sigma}")
    #prob_distribution = invert_normal_distribution(prob_distribution)
    if x==20:
        x=10
    if x ==50:
        x=5
    if x==80:
        x=5
    prob_distribution = create_probability_distribution(num_options=100, 
                                                        mean_option=50, 
                                                        sigma=x)
    ax2.scatter(range(100), prob_distribution, label=f"x={50}, std={x}")

ax.set_title('Different means', fontsize=15)
ax.set_ylabel('Probability')
ax.set_xlabel('Confidence')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
ax2.set_title('Different STD', fontsize=15)
ax2.set_xlabel('Confidence')
ax2.spines[['top', 'right']].set_visible(False)
ax2.legend()
plt.show()


