# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 00:10:22 2025

@author: carll
"""


#-----------------------------------------------------------------------------
# Model fitting functions
#-----------------------------------------------------------------------------
import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import TimeSeriesSplit

def normalize(params, bounds):
    return [np.clip((p - b[0]) / (b[1] - b[0]), 0, 1)
            for p, b in zip(params, bounds)]

def inverse_normalize(norm_params, bounds):
    return [np.clip(p * (b[1] - b[0]) + b[0], b[0], b[1])
            for p, b in zip(norm_params, bounds)]

def fit_model(model, args, bounds, n_trials, start_value_number=10,
              solver="L-BFGS-B", bias_model_best_params=False):

    # Normalize bounds for the optimization process
    norm_bounds = [(0, 1) for _ in bounds]

    # Wrap model function to inverse normalize parameters before evaluation
    def model_wrapper(norm_params, *args):
        params = inverse_normalize(norm_params, bounds)
        return model(params, *args) # Returns model function

    # Generate normalized start values
    start_ranges = []
    for bound in norm_bounds:
        lower_bound, upper_bound = bound
        start_range = np.random.uniform(lower_bound, upper_bound,
                                        size=start_value_number)
        start_ranges.append(start_range)

    # Optimization process
    result_list = []
    nll_list = []
    for i, norm_start_values in enumerate(zip(*start_ranges)):

        # Use best bias params for first run
        if i == 0 and bias_model_best_params!=False:
            norm_start_values = bias_model_best_params

        # Fit model to data
        result = minimize(model_wrapper, norm_start_values, args=args,
                          bounds=norm_bounds, method=solver)

        # Inverse normalize optimized parameters
        # to get them back in their original scale
        best_params = inverse_normalize(result.x, bounds)
        nll = result.fun

        # McFadden pseudo r2
        n_options = 101 # levels of confidence (i.e., 0,1,2... ...99, 100)
        nll_model = nll
        nll_null = -n_trials * np.log(1 / n_options)
        pseudo_r2 = 1 - (nll_model / nll_null)

        # AIC and BIC
        k = len(best_params)
        aic = 2*k + 2*nll
        bic = k*np.log(n_trials) + 2*nll

        # Append to result and nll lists
        result_list.append(best_params + [nll, aic, bic, pseudo_r2])
        nll_list.append(nll)

    # Index best result
    best_result_index = nll_list.index(min(nll_list))
    best_result = result_list[best_result_index]

    return best_result

def fit_model_with_LOO_cv(model, args, bounds, n_trials,
                      start_value_number=50,
                      solver="L-BFGS-B",
                      bias_model_best_params=False):

    # Normalize bounds for the optimization process
    norm_bounds = [(0, 1) for _ in bounds]

    # Leave-One-Out Cross-Validation setup
    loo = LeaveOneOut()

    # Placeholder for cross-validation results
    cv_results = []

    n_adjust = 1
    for train_index, test_index in loo.split(np.arange(n_trials-n_adjust)):

        # Wrap model function to inverse normalize parameters before evaluation
        def model_wrapper(norm_params, *args):
            params = inverse_normalize(norm_params, bounds)
            return model(params, *args) # Returns model function

        # Generate normalized start values
        start_ranges = []
        for bound in norm_bounds:
            lower_bound, upper_bound = bound
            start_range = np.random.uniform(lower_bound, upper_bound,
                                            size=start_value_number)
            start_ranges.append(start_range)

        # Optimization process
        result_list = []
        nll_list = []
        for i, norm_start_values in enumerate(zip(*start_ranges)):
            # Use best bias params for first run
            if i == 0 and bias_model_best_params != False:
                norm_start_values = bias_model_best_params

            # Fit model to data
            args_train = args + (train_index,) # Add index for training trials
            result = minimize(model_wrapper, norm_start_values,
                              args=args_train, bounds=norm_bounds,
                              method=solver)

            # Inverse normalize optimized parameters
            # to get them back in their original scale
            best_params = inverse_normalize(result.x, bounds)
            nll = result.fun

            # Evaluate on validation data
            args_test = args + (test_index,) # Add index for test trials
            val_nll = model(best_params, *args_test)
            #val_nll = model_wrapper(result.x, args_train)

            # McFadden pseudo r2
            n_options = 101 # levels of confidence (i.e., 0,1,2... ...99,100)
            nll_model = nll
            nll_null = -n_trials * np.log(1 / n_options)
            pseudo_r2 = 1 - (nll_model / nll_null)

            # AIC and BIC
            # OBS! As these are calculated using cross-validated NLL,
            # the AIC and BIC now only reflect a model complexity penalty after
            # overfitting have already been controlled for by the
            # cross-validation. Hence, the AIC and BIC measures should not be
            # thought of as a control for overfitting in this case.

            k = len(best_params)
            aic = 2*k + 2*val_nll
            bic = k*np.log(n_trials) + 2*val_nll

            # Append to result and nll lists
            result_list.append(best_params + [val_nll, aic, bic, pseudo_r2])
            nll_list.append(val_nll)

        # Index best result
        best_result_index = nll_list.index(min(nll_list))
        best_result = result_list[best_result_index]

        # Append the best result for this fold
        cv_results.append(best_result)

    # Average results over all folds
    averaged_result = np.mean(cv_results, axis=0)

    return averaged_result

def fit_model_with_cv(model, args, bounds, n_trials,
                      start_value_number=50,
                      solver="L-BFGS-B",
                      bias_model_best_params=False):

    """
    Uses time series cross-validation
    """

    # Normalize bounds for the optimization process
    norm_bounds = [(0, 1) for _ in bounds]

    # Placeholder for cross-validation results
    cv_results = []

    # Remove 2 trials:
    # The first trial in the session is not fit.
    # And another trial needs to be removed for the cross-validation.
    n_adjust_sess = 1 # 1 trial removed from session
    n_adjust_cv = 1 # 1 trial removed from cv splits
    n_trials_adjusted = n_trials-n_adjust_sess
    n_splits_cv = n_trials_adjusted-n_adjust_cv # Train on past, test on future.

    # Timeseries Cross-Validation setup
    tscv = TimeSeriesSplit(max_train_size=None, n_splits=n_splits_cv)

    for train_index, test_index in tscv.split(np.arange(n_trials_adjusted)):

        # Wrap model function to inverse normalize parameters before evaluation
        def model_wrapper(norm_params, *args):
            params = inverse_normalize(norm_params, bounds)
            return model(params, *args) # Returns model function

        # Generate normalized start values
        start_ranges = []
        for bound in norm_bounds:
            lower_bound, upper_bound = bound
            start_range = np.random.uniform(lower_bound, upper_bound,
                                            size=start_value_number)
            start_ranges.append(start_range)

        # Optimization process
        result_list = []
        nll_list = []
        for i, norm_start_values in enumerate(zip(*start_ranges)):
            # Use best bias params for first run
            if i == 0 and bias_model_best_params != False:
                norm_start_values = bias_model_best_params

            # Fit model to data
            args_train = args + (train_index,) # Add index for training trials
            result = minimize(model_wrapper, norm_start_values,
                              args=args_train, bounds=norm_bounds,
                              method=solver)

            # Inverse normalize optimized parameters
            # to get them back in their original scale
            best_params = inverse_normalize(result.x, bounds)
            nll = result.fun

            # Evaluate on validation data
            args_test = args + (test_index,) # Add index for test trials
            val_nll = model(best_params, *args_test)
            #val_nll = model_wrapper(result.x, args_train)

            # McFadden pseudo r2
            n_options = 101 # levels of confidence (i.e., 0,1,2... ...99,100)
            nll_model = nll
            nll_null = -n_trials * np.log(1 / n_options)
            pseudo_r2 = 1 - (nll_model / nll_null)

            # AIC and BIC
            # OBS! As these are calculated on cross-validated NLL,
            # the AIC and BIC now only reflect a model complexity penalty after
            # overfitting have already been controlled for by the
            # cross-validation. Hence, the AIC and BIC measures should not be
            # thought of as a control for overfitting in this case.

            k = len(best_params)
            aic = 2*k + 2*val_nll
            bic = k*np.log(n_trials) + 2*val_nll

            # Append to result and nll lists
            result_list.append(best_params + [val_nll, aic, bic, pseudo_r2])
            nll_list.append(val_nll)

        # Index best result
        best_result_index = nll_list.index(min(nll_list))
        best_result = result_list[best_result_index]

        # Append the best result for this fold
        cv_results.append(best_result)

    # Average results over all folds
    averaged_result = np.mean(cv_results, axis=0)

    return averaged_result