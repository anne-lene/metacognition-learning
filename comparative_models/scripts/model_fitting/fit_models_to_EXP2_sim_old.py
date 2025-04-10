# -*- coding: utf-8 -*-
"""
Created on Sat Mar 29 00:18:28 2025

@author: carll
"""

# ----------------------------------------------------------------------------
# Fit models to Simulated Data - Experiment 2: Variable Feedback

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
from src.models import (RWPD,
                        RWP,
                        RWCK,
                        CK,
                        RW_cond,
                        RW,
                        WSLS,
                        bias_model,
                        random_model)

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
    # For dot-estimate performance and feedback, we could use random noise, but
    # using participant performance and feedback has the advantage that it
    # tells us if we can recover models and parameters from the performance
    # and feedback values we know are in the data.
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

    # Biased confidence model
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

    # Win-stay-lose-shift model
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

    # Rescorla-Wagner model
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

    # Rescorla wagner Condition alpha model
    alpha_neut_bound = (0, 1)  # Alpha neut
    alpha_pos_bound =  (0, 1)  # Alpha pos
    alpha_neg_bound =  (0, 1)  # Alpha neg
    sigma_bound = (1, 10)    # Standard deviation
    bias_bound = (0, 100)     # Mean at first trial
    bounds = [(alpha_neut_bound[0], alpha_neut_bound[1]),
              (alpha_pos_bound[0], alpha_pos_bound[1]),
              (alpha_neg_bound[0], alpha_neg_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1])]
    results_rw_cond = fit_model_with_cv(model=RW_cond,
                                args=(confidence, feedback, n_trials,
                                      df_s.condition.values),
                                bounds=bounds,
                                n_trials=n_trials,
                                start_value_number=50,
                                solver="L-BFGS-B")
    alpha_neut_rw_cond = results_rw_cond[0].item()
    alpha_pos_rw_cond = results_rw_cond[1].item()
    alpha_neg_rw_cond = results_rw_cond[2].item()
    sigma_rw_cond = results_rw_cond[3].item()
    bias_rw_cond = results_rw_cond[4].item()
    nll_rw_cond = results_rw_cond[5].item()
    aic_rw_cond = results_rw_cond[6].item()
    bic_rw_cond = results_rw_cond[7].item()
    pseudo_r2_rw_cond = results_rw_cond[8].item()

    # Choice Kernel model
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

    # RW + Choice Kernel model
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

    # Rescorla-Wagner Performance model (RWP)
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    w_rw_bound = (0, 1)
    w_delta_p_bound = (0.2, 1)
    intercept_bound = (0, 100)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (w_rw_bound[0], w_rw_bound[1]),
              (w_delta_p_bound[0], w_delta_p_bound[1]),
              (intercept_bound[0], intercept_bound[1])
              ]
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
    w_rw_rwp = results_rwp[3].item()
    w_performance_rwp = results_rwp[4].item()
    intercept_rwp = results_rwp[5].item()
    nll_rwp = results_rwp[6].item()
    aic_rwp = results_rwp[7].item()
    bic_rwp = results_rwp[8].item()
    pseudo_r2_rwp = results_rwp[9].item()

    # Rescorla-Wagner Performance Delta model (RWPD)
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    bias_bound = (0, 100)
    w_rw_bound = (0, 1)
    w_delta_p_bound = (1, 3)
    intercept_bound = (0, 100)
    bounds = [(alpha_bound[0], alpha_bound[1]),
              (sigma_bound[0], sigma_bound[1]),
              (bias_bound[0], bias_bound[1]),
              (w_rw_bound[0], w_rw_bound[1]),
              (w_delta_p_bound[0], w_delta_p_bound[1]),
              (intercept_bound[0], intercept_bound[1]),
              ]
    results_rwpd = fit_model_with_cv(model=RWPD,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          performance),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=50,
                                    solver="L-BFGS-B")

    alpha_rwpd = results_rwpd[0].item()
    sigma_rwpd = results_rwpd[1].item()
    bias_rwpd = results_rwpd[2].item()
    w_rw_rwpd = results_rwpd[3].item()
    w_pd_rwpd = results_rwpd[4].item()
    intercept_rwpd = results_rwpd[5].item()
    nll_rwpd = results_rwpd[6].item()
    aic_rwpd = results_rwpd[7].item()
    bic_rwpd = results_rwpd[8].item()
    pseudo_r2_rwpd = results_rwpd[9].item()

    # Store session metrics
    # The p at the end of the key name stands for person as each value is
    # personal.
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
        'alpha_neut_rw_cond': alpha_neut_rw_cond,
        'alphaos_rw_cond': alpha_pos_rw_cond,
        'alpha_neg_rw_cond': alpha_neg_rw_cond,
        'sigma_rw_cond': sigma_rw_cond,
        'bias_rw_cond': bias_rw_cond,
        'nll_rw_cond': nll_rw_cond,
        'aic_rw_cond': aic_rw_cond,
        'bic_rw_cond': bic_rw_cond,
        'pseudo_r2_rw_cond': pseudo_r2_rw_cond,
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
        'w_rw_rwp': w_rw_rwp,
        'w_performance_rwp': w_performance_rwp,
        'intercept_rwp': intercept_rwp,
        'nll_rwp': nll_rwp,
        'aic_rwp': aic_rwp,
        'bic_rwp': bic_rwp,
        'pseudo_r2_rwp': pseudo_r2_rwp,
        'alpha_rwpd': alpha_rwpd,
        'sigma_rwpd': sigma_rwpd,
        'bias_rwpd': bias_rwpd,
        'w_rw_rwpd': w_rw_rwpd,
        'w_pd_rwpd': w_pd_rwpd,
        'intercept_rwpd': intercept_rwpd,
        'nll_rwpd': nll_rwpd,
        'aic_rwpd': aic_rwpd,
        'bic_rwpd': bic_rwpd,
        'pseudo_r2_rwpd': pseudo_r2_rwpd,
    }

    return pid_session_model_tuple, session_metrics

#-----------------------------------------------------------------------------
# Main function to perform model fitting on EXP2 data
#-----------------------------------------------------------------------------

def main(df):

    # Unique pid, model, and sessions pairs
    unique_pairs = df[['pid', 'session', 'model']].drop_duplicates()

    # Convert the DataFrame to a list of tuples
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

    # Create a list of DataFrames, one for each unique pid-model-session combo
    df_list = [df[(df['pid'] == pid) &
                  (df['session'] == session) &
                  (df['model'] == model)].copy()
               for pid, session, model in unique_pairs_list]

    # Total number of unique combos
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
        'alpha_neut_rw_cond': np.zeros(num_unique_combos),
        'alphaos_rw_cond': np.zeros(num_unique_combos),
        'alpha_neg_rw_cond': np.zeros(num_unique_combos),
        'sigma_rw_cond': np.zeros(num_unique_combos),
        'bias_rw_cond': np.zeros(num_unique_combos),
        'nll_rw_cond': np.zeros(num_unique_combos),
        'aic_rw_cond': np.zeros(num_unique_combos),
        'bic_rw_cond': np.zeros(num_unique_combos),
        'pseudo_r2_rw_cond': np.zeros(num_unique_combos),
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
        'w_rw_rwp': np.zeros(num_unique_combos),
        'w_performance_rwp': np.zeros(num_unique_combos),
        'intercept_rwp': np.zeros(num_unique_combos),
        'nll_rwp': np.zeros(num_unique_combos),
        'aic_rwp': np.zeros(num_unique_combos),
        'bic_rwp': np.zeros(num_unique_combos),
        'pseudo_r2_rwp': np.zeros(num_unique_combos),
        'alpha_rwpd': np.zeros(num_unique_combos),
        'sigma_rwpd': np.zeros(num_unique_combos),
        'bias_rwpd': np.zeros(num_unique_combos),
        'w_rw_rwpd': np.zeros(num_unique_combos),
        'w_pd_rwpd': np.zeros(num_unique_combos),
        'intercept_rwpd': np.zeros(num_unique_combos),
        'nll_rwpd': np.zeros(num_unique_combos),
        'aic_rwpd': np.zeros(num_unique_combos),
        'bic_rwpd': np.zeros(num_unique_combos),
        'pseudo_r2_rwpd': np.zeros(num_unique_combos),
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
    relative_path = "../../results/variable_feedback/model_comparison/model_and_param_recovery"

    # Construct the full path to the file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))
    file_name = r"model_fits_EXP2_sim"
    save_path = os.path.join(relative_path, file_path, file_name)

    # Construct and save dataframe
    df_m = pd.DataFrame(results)
    df_m.to_excel(f"{save_path}.xlsx") # save to specified directory

#-----------------------------------------------------------------------------
# Execution
#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Import data - Varied feedback (Experiment 2)
    df = load_sim_df(EXP=2)

    # For debugging
    #df = df[df['pid']=='5c884b29c2ceec001719b1e4']

    # Run main
    main(df)


