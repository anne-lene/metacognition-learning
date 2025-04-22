# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 16:00:58 2025

@author: carll
"""

# ----------------------------------------------------------------------------
# Fit models to Simulated Data - Experiment 1: Fixed Feedback

# Saves result in:
# ../../results/variable_feedback/model_comparison/model_and_param_recovery
# ----------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from src.model_fitting_functions import fit_model_with_cv
from src.utils import load_sim_df
from src.models import (random_model,
                        bias_model,
                        RWFP,
                        RWP,
                        RWCK,
                        CK,
                        RW,
                        WSLS,
                        LMF,
                        LMP,
                        LMFP
                        )

# Function to process data for a single session
def process_session(df_s):

    # Participant and session and model
    participant = df_s.pid.unique()[0]
    session = df_s.session.unique()[0]
    model_simulated = df_s.model.unique()[0]
    pid_session_model_tuple = (participant, session, model_simulated)

    # Remove baseline
    df_s = df_s[df_s.condition != 'baseline']

    # Calculate absolute trial error as the average across subtrials
    df_s['difference'] = abs(df_s['estimate'] - df_s['correct'])
    abs_error_avg = df_s.groupby('trial')['difference'].mean()

    # Only keep first row of every subtrial (one row = one trial)
    df_s = df_s.drop_duplicates(subset='trial', keep='first')

    # For debugging
    #df_s = df_s.head(5)

    # Condition
    condition = df_s.condition.unique()[0]

    # N trials
    n_trials = len(df_s)

    # Calculate trial-by-trial metrics
    confidence = df_s.confidence_sim.values # Simulated estimates.
    feedback = df_s.feedback.values # Feedback from participant.
    performance = -abs_error_avg.values # Performance from participant.

    # Random confidence model
    bounds = [(0, 0)]
    results = fit_model_with_cv(model=random_model,
                                args=(101, n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")

    nll_random = results[1].item()
    aic_random = results[2].item()
    bic_random = results[3].item()
    pseudo_r2_random = results[4].item()

    # Bias: Biased confidence model
    mean_bound = (0, 100)
    sigma_bound = (1, 10)
    bounds = [(mean_bound[0], mean_bound[1]),
              (sigma_bound[0], sigma_bound[1])]
    results = fit_model_with_cv(model=bias_model,
                                args=(confidence, n_trials),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")
    mean_bias = results[0].item()
    sd_bias= results[1].item()
    nll_bias = results[2].item()
    aic_bias = results[3].item()
    bic_bias = results[4].item()
    pseudo_r2_bias = results[5].item()

    # WSLS: Win-Stay-Lose-Shift model
    sigma_bound = (1, 10)
    win_bound = (1, 100)
    bounds = [(sigma_bound[0], sigma_bound[1]),
              (win_bound[0], win_bound[1])]
    results_win_stay = fit_model_with_cv(model=WSLS,
                                         args=(confidence,
                                               feedback,
                                               n_trials),
                                         bounds=bounds,
                                         n_trials=n_trials,
                                         start_value_number=50,
                                         solver="L-BFGS-B")

    std_WSLS_array = results_win_stay[0].item()
    win_boundary_WSLS_array = results_win_stay[1].item()
    nll_win_stay = results_win_stay[2].item()
    aic_win_stay = results_win_stay[3].item()
    bic_win_stay = results_win_stay[4].item()
    pseudo_r2_win_stay = results_win_stay[5].item()

    # RW: Rescorla-Wagner model
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]
    results_rw_symm = fit_model_with_cv(model=RW,
                                        args=(confidence, feedback, n_trials),
                                        bounds=bounds, n_trials=n_trials,
                                        start_value_number=50,
                                        solver="L-BFGS-B")
    alpha_rw_symm = results_rw_symm[0].item()
    sigma_rw_symm = results_rw_symm[1].item()
    bias_rw_symm = results_rw_symm[2].item()
    nll_rw_symm = results_rw_symm[3].item()
    aic_rw_symm = results_rw_symm[4].item()
    bic_rw_symm = results_rw_symm[5].item()
    pseudo_r2_rw_symm = results_rw_symm[6].item()

    # CK: Choice Kernel model
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    beta_bound = (40, 200)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1])]
    results_ck = fit_model_with_cv(model=CK,
                                   args=(confidence, n_trials),
                                   bounds=bounds,
                                   n_trials=n_trials,
                                   start_value_number=50,
                                   solver="L-BFGS-B")
    alpha_ck = results_ck[0].item()
    sigma_ck = results_ck[1].item()
    bias_ck = results_ck[2].item()
    beta_ck = results_ck[3].item()
    nll_ck = results_ck[4].item()
    aic_ck = results_ck[5].item()
    bic_ck = results_ck[6].item()
    pseudo_r2_ck = results_ck[7].item()

    # RWCK: Rescorla-Wagner + Choice Kernel model
    alpha_bound = (0, 1)
    alpha_ck_bound = (0, 1)
    sigma_bound = (1, 10)
    sigma_ck_bound = (1, 10)
    bias_bound = (0, 100)
    beta_bound = (40, 200)
    beta_ck_bound = (40, 200)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (alpha_ck_bound[0], alpha_ck_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (sigma_ck_bound[0], sigma_ck_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (beta_bound[0], beta_bound[1]),
              (beta_ck_bound[0], beta_ck_bound[1])]
    results_rwck = fit_model_with_cv(model=RWCK,
                                     args=(feedback, confidence, n_trials),
                                     bounds=bounds,
                                     n_trials=n_trials,
                                     start_value_number=50,
                                     solver="L-BFGS-B")

    alpha_rwck = results_rwck[0].item()
    alpha_ck_rwck = results_rwck[1].item()
    sigma_rwck = results_rwck[2].item()
    sigma_ck_rwck = results_rwck[3].item()
    bias_rwck = results_rwck[4].item()
    beta_rwck = results_rwck[5].item()
    ck_beta_rwck = results_rwck[6].item()
    nll_rwck = results_rwck[7].item()
    aic_rwck = results_rwck[8].item()
    bic_rwck = results_rwck[9].item()
    pseudo_r2_rwck = results_rwck[10].item()

    # RWP: Rescorla-Wagner Performance model
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    wp_bound = (0, 2)

    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (wp_bound[0], wp_bound[1])]

    results_rwp = fit_model_with_cv(model=RWP,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

    alpha_rwp = results_rwp[0].item()
    sigma_rwp = results_rwp[1].item()
    bias_rwp = results_rwp[2].item()
    wp_rwp = results_rwp[3].item()
    nll_rwp = results_rwp[4].item()
    aic_rwp = results_rwp[5].item()
    bic_rwp = results_rwp[6].item()
    pseudo_r2_rwp = results_rwp[7].item()

    # RWFP: Rescorla-Wagner Feedback and Performance model
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    wf_bound = (0, 2)
    wp_bound = (0, 2)

    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (wf_bound[0], wf_bound[1]),
              (wp_bound[0], wp_bound[1])]

    results_rwfp = fit_model_with_cv(model=RWFP,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

    alpha_rwfp = results_rwfp[0].item()
    sigma_rwfp = results_rwfp[1].item()
    bias_rwfp = results_rwfp[2].item()
    wp_rwfp = results_rwfp[3].item()
    wf_rwfp = results_rwfp[4].item()
    nll_rwfp = results_rwfp[5].item()
    aic_rwfp = results_rwfp[6].item()
    bic_rwfp = results_rwfp[7].item()
    pseudo_r2_rwfp = results_rwfp[8].item()

    # LMF: Linear model of Feedback.
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    intercept_bound = (0, 100)
    wf_bound = (0, 2)

    bounds = [(sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (intercept_bound[0], intercept_bound[1]),
              (wf_bound[0], wf_bound[1])]


    results_lmf = fit_model_with_cv(model=LMF,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

    sigma_lmf = results_lmf[0].item()
    bias_lmf = results_lmf[1].item()
    intercept_lmf = results_lmf[2].item()
    wf_lmf = results_lmf[3].item()
    nll_lmf = results_lmf[4].item()
    aic_lmf = results_lmf[5].item()
    bic_lmf = results_lmf[6].item()
    pseudo_r2_lmf = results_lmf[7].item()

    # LMP: Linear model of Performance.
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    intercept_bound = (0, 100)
    wp_bound = (0, 2)

    bounds = [(sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (intercept_bound[0], intercept_bound[1]),
              (wp_bound[0], wp_bound[1])]


    results_lmp = fit_model_with_cv(model=LMP,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

    sigma_lmp = results_lmp[0].item()
    bias_lmp = results_lmp[1].item()
    intercept_lmp = results_lmp[2].item()
    wp_lmp = results_lmp[3].item()
    nll_lmp = results_lmp[4].item()
    aic_lmp = results_lmp[5].item()
    bic_lmp = results_lmp[6].item()
    pseudo_r2_lmp = results_lmp[7].item()

    # LMFP: Linear model of Feedback and Performance.
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    intercept_bound = (0, 100)
    wf_bound = (0, 1)
    wp_bound = (0, 1)

    bounds = [(sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (intercept_bound[0], intercept_bound[1]),
              (wf_bound[0], wf_bound[1]),
              (wp_bound[0], wp_bound[1])]


    results_lmfp = fit_model_with_cv(model=LMFP,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

    sigma_lmfp = results_lmfp[0].item()
    bias_lmfp = results_lmfp[1].item()
    intercept_lmfp = results_lmfp[2].item()
    wf_lmfp = results_lmfp[3].item()
    wp_lmfp = results_lmfp[4].item()
    nll_lmfp = results_lmfp[5].item()
    aic_lmfp = results_lmfp[6].item()
    bic_lmfp = results_lmfp[7].item()
    pseudo_r2_lmfp = results_lmfp[8].item()

    # Store session metrics
    session_metrics = {
        'pid': participant,
        'session': session,
        'model': model_simulated,
        'condition': condition,
        'n_trials': n_trials,
        'nll_random': nll_random,
        'aic_random': aic_random,
        'bic_random': bic_random,
        'pseudo_r2_random': pseudo_r2_random,
        'mean_bias': mean_bias,
        'sd_bias': sd_bias,
        'nll_bias': nll_bias,
        'aic_bias': aic_bias,
        'bic_bias': bic_bias,
        'pseudo_r2_bias': pseudo_r2_bias,
        'std_WSLS': std_WSLS_array,
        'win_boundary_WSLS': win_boundary_WSLS_array,
        'nll_win_stay': nll_win_stay,
        'aic_win_stay': aic_win_stay,
        'bic_win_stay': bic_win_stay,
        'pseudo_r2_win_stay': pseudo_r2_win_stay,
        'alpha_rw_symm': alpha_rw_symm,
        'sigma_rw_symm': sigma_rw_symm,
        'bias_rw_symm': bias_rw_symm,
        'nll_rw_symm': nll_rw_symm,
        'aic_rw_symm': aic_rw_symm,
        'bic_rw_symm': bic_rw_symm,
        'pseudo_r2_rw_symm': pseudo_r2_rw_symm,
        'alpha_ck': alpha_ck,
        'sigma_ck': sigma_ck,
        'bias_ck': bias_ck,
        'beta_ck': beta_ck,
        'nll_ck': nll_ck,
        'aic_ck': aic_ck,
        'bic_ck': bic_ck,
        'pseudo_r2_ck': pseudo_r2_ck,
        'alpha_rwck': alpha_rwck,
        'alpha_ck_rwck': alpha_ck_rwck,
        'sigma_rwck': sigma_rwck,
        'sigma_ck_rwck': sigma_ck_rwck,
        'bias_rwck': bias_rwck,
        'beta_rwck': beta_rwck,
        'ck_beta_rwck': ck_beta_rwck,
        'nll_rwck': nll_rwck,
        'aic_rwck': aic_rwck,
        'bic_rwck': bic_rwck,
        'pseudo_r2_rwck': pseudo_r2_rwck,
        'alpha_rwp': alpha_rwp,
        'sigma_rwp': sigma_rwp,
        'bias_rwp': bias_rwp,
        'wp_rwp': wp_rwp,
        'nll_rwp': nll_rwp,
        'aic_rwp': aic_rwp,
        'bic_rwp': bic_rwp,
        'pseudo_r2_rwp': pseudo_r2_rwp,
        'alpha_rwfp': alpha_rwfp,
        'sigma_rwfp': sigma_rwfp,
        'bias_rwfp': bias_rwfp,
        'wf_rwfp': wf_rwfp,
        'wp_rwfp': wp_rwfp,
        'nll_rwfp': nll_rwfp,
        'aic_rwfp': aic_rwfp,
        'bic_rwfp': bic_rwfp,
        'pseudo_r2_rwfp': pseudo_r2_rwfp,
        'sigma_lmf': sigma_lmf,
        'bias_lmf': bias_lmf,
        'intercept_lmf': intercept_lmf,
        'wf_lmf': wf_lmf,
        'nll_lmf': nll_lmf,
        'aic_lmf': aic_lmf,
        'bic_lmf': bic_lmf,
        'pseudo_r2_lmf': pseudo_r2_lmf,
        'sigma_lmp': sigma_lmp,
        'bias_lmp': bias_lmp,
        'intercept_lmp': intercept_lmp,
        'wp_lmp': wp_lmp,
        'nll_lmp': nll_lmp,
        'aic_lmp': aic_lmp,
        'bic_lmp': bic_lmp,
        'pseudo_r2_lmp': pseudo_r2_lmp,
        'sigma_lmfp': sigma_lmfp,
        'bias_lmfp': bias_lmfp,
        'intercept_lmfp': intercept_lmfp,
        'wf_lmfp': wf_lmfp,
        'wp_lmfp': wp_lmfp,
        'nll_lmfp': nll_lmfp,
        'aic_lmfp': aic_lmfp,
        'bic_lmfp': bic_lmfp,
        'pseudo_r2_lmfp': pseudo_r2_lmfp,
    }

    return pid_session_model_tuple, session_metrics

#-----------------------------------------------------------------------------
# Main function to perform model fitting on EXP1 sim
#-----------------------------------------------------------------------------

def main(df):

    # Remove baseline trials before computing combinations
    df = df[df['condition'] != 'baseline']

    # Unique pid, model, and sessions pairs
    unique_pairs = df[['pid', 'session', 'model']].drop_duplicates()

    # Convert the DataFrame to a list of tuples
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))
    unique_pairs_list = unique_pairs_list[:300]
    # Create a list of DataFrames, one for each unique pid-model-session combo
    df_list = [df[(df['pid'] == pid) &
                  (df['session'] == session) &
                  (df['model'] == model)].copy()
               for pid, session, model in unique_pairs_list]

    # Print job summary
    print('df_list len:', len(df_list))
    num_unique_combos = len(unique_pairs_list)
    print('PIDs:', len(df['pid'].unique()))
    print('Sessions:', len(df['session'].unique()))
    print('Models:', len(df['model'].unique()))
    print(f"Number of pid-model-session combos: {num_unique_combos}")

    # Initialize result dictionaries for all participants
    results = {
        # Using dtype=object for strings
        'pid': np.empty(num_unique_combos, dtype=object),
        'session': np.zeros(num_unique_combos),
        'model': np.zeros(num_unique_combos, dtype=object), # Simulated model
        'condition':  np.empty(num_unique_combos, dtype=object),
        'n_trials': np.zeros(num_unique_combos),
        'nll_random': np.zeros(num_unique_combos),
        'aic_random': np.zeros(num_unique_combos),
        'bic_random': np.zeros(num_unique_combos),
        'pseudo_r2_random': np.zeros(num_unique_combos),
        'mean_bias': np.zeros(num_unique_combos),
        'sd_bias': np.zeros(num_unique_combos),
        'nll_bias': np.zeros(num_unique_combos),
        'aic_bias': np.zeros(num_unique_combos),
        'bic_bias': np.zeros(num_unique_combos),
        'pseudo_r2_bias': np.zeros(num_unique_combos),
        'std_WSLS': np.zeros(num_unique_combos),
        'win_boundary_WSLS': np.zeros(num_unique_combos),
        'nll_win_stay': np.zeros(num_unique_combos),
        'aic_win_stay': np.zeros(num_unique_combos),
        'bic_win_stay': np.zeros(num_unique_combos),
        'pseudo_r2_win_stay': np.zeros(num_unique_combos),
        'alpha_rw_symm': np.zeros(num_unique_combos),
        'sigma_rw_symm': np.zeros(num_unique_combos),
        'bias_rw_symm': np.zeros(num_unique_combos),
        'nll_rw_symm': np.zeros(num_unique_combos),
        'aic_rw_symm': np.zeros(num_unique_combos),
        'bic_rw_symm': np.zeros(num_unique_combos),
        'pseudo_r2_rw_symm': np.zeros(num_unique_combos),
        'alpha_ck': np.zeros(num_unique_combos),
        'sigma_ck': np.zeros(num_unique_combos),
        'bias_ck': np.zeros(num_unique_combos),
        'beta_ck': np.zeros(num_unique_combos),
        'nll_ck': np.zeros(num_unique_combos),
        'aic_ck': np.zeros(num_unique_combos),
        'bic_ck': np.zeros(num_unique_combos),
        'pseudo_r2_ck': np.zeros(num_unique_combos),
        'alpha_rwck': np.zeros(num_unique_combos),
        'alpha_ck_rwck': np.zeros(num_unique_combos),
        'sigma_rwck': np.zeros(num_unique_combos),
        'sigma_ck_rwck': np.zeros(num_unique_combos),
        'bias_rwck': np.zeros(num_unique_combos),
        'beta_rwck': np.zeros(num_unique_combos),
        'ck_beta_rwck': np.zeros(num_unique_combos),
        'nll_rwck': np.zeros(num_unique_combos),
        'aic_rwck': np.zeros(num_unique_combos),
        'bic_rwck': np.zeros(num_unique_combos),
        'pseudo_r2_rwck': np.zeros(num_unique_combos),
        'alpha_rwp': np.zeros(num_unique_combos),
        'sigma_rwp': np.zeros(num_unique_combos),
        'bias_rwp': np.zeros(num_unique_combos),
        'wp_rwp': np.zeros(num_unique_combos),
        'nll_rwp': np.zeros(num_unique_combos),
        'aic_rwp': np.zeros(num_unique_combos),
        'bic_rwp': np.zeros(num_unique_combos),
        'pseudo_r2_rwp': np.zeros(num_unique_combos),
        'alpha_rwfp': np.zeros(num_unique_combos),
        'sigma_rwfp': np.zeros(num_unique_combos),
        'bias_rwfp': np.zeros(num_unique_combos),
        'wf_rwfp': np.zeros(num_unique_combos),
        'wp_rwfp': np.zeros(num_unique_combos),
        'nll_rwfp': np.zeros(num_unique_combos),
        'aic_rwfp': np.zeros(num_unique_combos),
        'bic_rwfp': np.zeros(num_unique_combos),
        'pseudo_r2_rwfp': np.zeros(num_unique_combos),
        'sigma_lmf': np.zeros(num_unique_combos),
        'bias_lmf': np.zeros(num_unique_combos),
        'intercept_lmf': np.zeros(num_unique_combos),
        'wf_lmf': np.zeros(num_unique_combos),
        'nll_lmf': np.zeros(num_unique_combos),
        'aic_lmf': np.zeros(num_unique_combos),
        'bic_lmf': np.zeros(num_unique_combos),
        'pseudo_r2_lmf': np.zeros(num_unique_combos),
        'sigma_lmp': np.zeros(num_unique_combos),
        'bias_lmp': np.zeros(num_unique_combos),
        'intercept_lmp': np.zeros(num_unique_combos),
        'wp_lmp': np.zeros(num_unique_combos),
        'nll_lmp': np.zeros(num_unique_combos),
        'aic_lmp': np.zeros(num_unique_combos),
        'bic_lmp': np.zeros(num_unique_combos),
        'pseudo_r2_lmp': np.zeros(num_unique_combos),
        'sigma_lmfp':  np.zeros(num_unique_combos),
        'bias_lmfp':  np.zeros(num_unique_combos),
        'intercept_lmfp': np.zeros(num_unique_combos),
        'wf_lmfp': np.zeros(num_unique_combos),
        'wp_lmfp': np.zeros(num_unique_combos),
        'nll_lmfp': np.zeros(num_unique_combos),
        'aic_lmfp': np.zeros(num_unique_combos),
        'bic_lmfp': np.zeros(num_unique_combos),
        'pseudo_r2_lmfp': np.zeros(num_unique_combos),
    }

    # Use multiprocessing to process each participant in parallel
    with Pool(48) as pool:
        for sim_id, metrics in tqdm(pool.map(process_session, df_list),
                                        total=len(df_list)):

            # Get the index of the current session
            idx = unique_pairs_list.index(sim_id)

            # Process each key in the metrics dictionary
            for key, value in metrics.items():
                print(key, idx, value)
                if key in ['pid', 'model','condition']:  # String fields
                    results[key][idx] = value
                elif hasattr(value, 'item'):
                        # Extract scalar if it's a NumPy object
                        results[key][idx] = value.item()
                else:
                    # Use directly if it's already a Python scalar
                    results[key][idx] = value

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative save path
    relative_path = "../../results/Fixed_feedback/model_comparison/model_and_param_recovery"

    # Construct the full path to the file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))
    file_name = r"model_fits_EXP1_sim"
    save_path = os.path.join(relative_path, file_path, file_name)

    # Construct and save dataframe
    df_m = pd.DataFrame(results)
    df_m.to_excel(f"{save_path}.xlsx") # save to specified directory

#-----------------------------------------------------------------------------
# Execution
#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Import data - Fixed feedback (Experiment 1)
    df = load_sim_df(EXP=1)

    # For debugging
    #df = df[df['pid'] == '55a239e0fdf99b3cec08f262']

    # Run main
    main(df)


