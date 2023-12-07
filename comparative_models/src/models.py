# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:26:08 2023

@author: carll
"""

# Models 

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

#%%



def rw_symmetric_LR(x, *args):
    
    """
    The symmetric learning rate model uses a Rescorla-Wagner rule to 
    update confidence, using the discrepancy between confidence and 
    performance feedback (fb) as the prediction error (PE).
    """
    
    alpha, sigma = x
    confidence, feedback, n_trials = args
    
    prediction_range = 100
    nll_vector = []
    for t in range(n_trials):
        
    
        if t == 0:
           
            log_likelihood = math.log((1/(prediction_range)))
            nll_vector.append(-log_likelihood)
            
        else: 
         
            # Get previous confidence value and feedback, 
            f = feedback[t-1]
            c = int(confidence[t-1])-1
            if c >= 100:
                c = int(f"{c}"[:2])
            PE = f - c # prediction error
            c_pred = c + alpha*PE
            
            if c_pred >= 100:
                c_pred = 100
            if c_pred <= 0:
                c_pred = 1
                
            # Probability distribution for win trials
            prob_distribution = create_probability_distribution(
                                            num_options=prediction_range, 
                                            mean_option=c_pred, 
                                            sigma=sigma)

            
            # Get negative log likelihood of confidence
            epsilon = 1e-10
            log_likelihood = math.log(prob_distribution[c] + epsilon)
            nll_vector.append(-log_likelihood)
   
    return sum(nll_vector)


def fit_rw_symmetric_LR(confidence, feedback, n_trials):
    x0 = [0.01, 50] # mean_option, sigma, win_boundary
    bounds = [(0.0001, 1), (10, 11)]
    result = minimize(rw_symmetric_LR, x0, args=(confidence,
                                                     feedback, 
                                                     n_trials), 
                      bounds=bounds, 
                      method="L-BFGS-B")
    
    best_alpha = result.x[0]
    best_std = result.x[1]
    
    nll = result.fun
    
    # McFadden pseudo r2
    nll_model = nll
    n = n_trials
    nll_null = -n * np.log(0.5)
    pseudo_r2 = 1 - (nll_model / nll_null)
    
    k = len(x0)
    aic = 2*k + 2*nll
    bic = k*np.log(n) + 2*nll
    
    return [best_alpha, best_std, nll, aic, bic, pseudo_r2]



def invert_normal_distribution(probabilities):
    
    """
    Ivert probability distribution, while keeping sum to 1.
    """
    # Find the maximum probability
    max_prob = np.max(probabilities)  
    # Invert the probabilities
    inverted = max_prob - probabilities  
    # Normalize the inverted probabilities
    normalized_inverted = inverted / np.sum(inverted)  
    
    return normalized_inverted


def win_stay_lose_shift(x, *args):
    
    """
    The negative log likelihood (NLL) when changing confidence when
    confidence and feedback does not overlap.
    """
    
    mean_option, sigma, win_boundary = x
    confidence, feedback, n_trials = args
    
    prediction_range = 100
    nll_vector = []
    for t in range(n_trials):
        
        epsilon = 1e-10
        if t == 0:
           
            
            log_likelihood = math.log((1/(prediction_range)))
            nll_vector.append(-log_likelihood)
            
        else: 
         
            # Get previous confidence value and set 
            f = feedback[t-1]
            c = int(confidence[t-1])-1
            if c >= 100:
                c = int(f"{c}"[:2])
            upper_bound = c + win_boundary
            lower_bound = c - win_boundary
            
            if f > lower_bound and f < upper_bound:
                
                # Probability distribution for win trials
                prob_distribution = create_probability_distribution(
                                                num_options=prediction_range, 
                                                mean_option=c, 
                                                sigma=sigma)
            
                
            else:
                
               
                # Probability distribution for lose trials
                prob_distribution = create_probability_distribution(
                                                num_options=prediction_range, 
                                                mean_option=c, 
                                                sigma=sigma)
                
                # Invert probabilities
                prob_distribution = invert_normal_distribution(
                                                            prob_distribution)
                
            # Get negative log likelihood of confidence
            log_likelihood = math.log(prob_distribution[c] + epsilon)
            nll_vector.append(-log_likelihood)
    
   
    return sum(nll_vector)



def fit_win_stay_lose_shift(confidence, feedback, n_trials):
    x0 = [50, 50, 50] # mean_option, sigma, win_boundary
    bounds = [(1, 99), (10, 11), (1, 100)]
    result = minimize(win_stay_lose_shift, x0, args=(confidence,
                                                     feedback, 
                                                     n_trials), 
                      bounds=bounds, 
                      method="L-BFGS-B")
    
    best_mean = result.x[0]
    best_std = result.x[1]
    best_win_boundary = result.x[2]
    
    nll = result.fun
    
    # McFadden pseudo r2
    nll_model = nll
    n = n_trials
    nll_null = -n * np.log(0.5)
    pseudo_r2 = 1 - (nll_model / nll_null)
    
    k = len(x0)
    aic = 2*k + 2*nll
    bic = k*np.log(n) + 2*nll
    
    return [best_mean, best_std, best_win_boundary, nll, aic, bic, pseudo_r2]



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



def random_model_w_bias(x, *args):
    
    """
    The negative log likelihood (NLL) of updating confidence 
    with a bias towards a specific range of options.
    """
    
    mean_option, sigma  = x
    confidence, n_trials = args
    
    prediction_range = 100
    nll_vector = []
    for t in range(n_trials):
        
        # Probability distribution 
        prob_distribution = create_probability_distribution(
                                                num_options=prediction_range, 
                                                mean_option=mean_option, 
                                                sigma=sigma)
        c = int(confidence[t])-1
        if c >= 100:
            c = int(f"{c}"[:2])-1
                  
      
       # print(c)
       # print(prob_distribution)
       # print(prob_distribution[c])
        epsilon = 1e-10
        log_likelihood = math.log(prob_distribution[c] + epsilon)
        nll_vector.append(-log_likelihood)
    

    return sum(nll_vector)


def fit_random_model_w_bias(confidence, n_trials):
    x0 = [50, 10,] # mean and std
    bounds = [(1, 99), (10, 11)]
    result = minimize(random_model_w_bias, x0, args=(confidence, n_trials), 
                      bounds=bounds, 
                      method="L-BFGS-B")
    best_mean = result.x[0]
    best_std = result.x[1]
    
    nll = result.fun
    
    # McFadden pseudo r2
    nll_model = nll
    n = n_trials
    nll_null = -n * np.log(0.5)
    pseudo_r2 = 1 - (nll_model / nll_null)
    
    k = len(x0)
    aic = 2*k + 2*nll
    bic = k*np.log(n) + 2*nll
    
    return [best_mean, best_std, nll, aic, bic, pseudo_r2]

def fit_random_model(prediction_range,
                        n_trials):
    
    """
    The negative log likelihood (NLL) of updating confidence randomly.
    """
    
    random_choice_probability = 1/prediction_range
    log_likelihood = math.log(random_choice_probability)
    nll_vector = [-log_likelihood]*n_trials
    
    return sum(nll_vector)









