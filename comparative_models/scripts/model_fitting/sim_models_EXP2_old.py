# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 02:44:41 2024

@author: carll
"""

# Simulating models on paramters drawn randomly from best-fit parameter range
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import beta
from src.utils import (add_session_column, load_df)
from src.models import (RWPD_sim,
                        RWP_sim,
                        RWCK_sim,
                        CK_sim,
                        RW_cond_sim,
                        RW_sim,
                        WSLS_sim,
                        bias_model_sim,
                        )

def process_session(df_s):

    # Participant and session
    participant = df_s.pid.unique()[0]
    session = df_s.session.unique()[0]

    # Remove baseline
    df_s = df_s[df_s.condition != 'baseline']

    # Calculate absolute trial error as the average across subtrials
    df_s['difference'] = abs(df_s['estimate'] - df_s['correct'])
    abs_error_avg = df_s.groupby('trial')['difference'].mean()


    # Only keep first row of every subtrial (one row = one trial)
    df_s = df_s.drop_duplicates(subset='trial', keep='first')

    # Ensure performance aligns with filtered df_s
    # performance is inverse of abs error
    df_s['performance'] = -abs_error_avg.values # df_s['trial'].map(lambda t: -abs_error_avg.loc[t])

    # Condition
    conditions = df_s.condition.unique()

    # N trials
    n_trials = len(df_s)

    # Calculate trial-by-trial metrics
    confidence = df_s.confidence.values
    feedback = df_s.feedback.values
    performance = -abs_error_avg.values

    # Initialise dict in which simulation results will be saved
    participant_results = {}

    # Bias model parameters and simulation
    mean_bound = (0, 100)
    sigma_bound = (1, 10)
    bias_params = {
        'mean': np.random.uniform(*mean_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }
    bias_sim_conf = bias_model_sim((bias_params['mean'], bias_params['sigma']), n_trials)
    participant_results.update({'bias_conf': bias_sim_conf, **{f'bias_{k}': v for k, v in bias_params.items()}})

    # WSLS model parameters and simulation
    sigma_bound = (1, 10)
    win_bound = (1, 100)
    wsls_params = {
        'sigma': np.random.uniform(*sigma_bound),
        'win_boundary': np.random.uniform(*win_bound)
    }
    wsls_sim_conf = WSLS_sim((wsls_params['sigma'], wsls_params['win_boundary']), confidence, feedback, n_trials)
    participant_results.update({'wsls_conf': wsls_sim_conf, **{f'wsls_{k}': v for k, v in wsls_params.items()}})

    # RW model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    rw_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }

    rw_sim_conf, y_val_rw = RW_sim((rw_params['alpha'], rw_params['sigma']), confidence, feedback, n_trials)
    participant_results.update({'rw_conf': rw_sim_conf, **{f'rw_{k}': v for k, v in rw_params.items()}})

    # RW cond model parameters and simulation
    alpha_neut_bound = (0, 1)
    alpha_pos_bound = (0, 1)
    alpha_neg_bound = (0, 1)
    sigma_bound = (1, 10)
    rw_cond_params = {
        'alpha_neut': np.random.uniform(*alpha_neut_bound),
        'alpha_pos': np.random.uniform(*alpha_pos_bound),
        'alpha_neg': np.random.uniform(*alpha_neg_bound),
        'sigma': np.random.uniform(*sigma_bound)
    }

    rw_cond_sim_conf = RW_cond_sim((rw_cond_params['alpha_neut'], rw_cond_params['alpha_pos'], rw_cond_params['alpha_neg'], rw_cond_params['sigma']), confidence, feedback, n_trials, df_s.condition.values)
    participant_results.update({'rw_cond_conf': rw_cond_sim_conf, **{f'rw_cond_{k}': v for k, v in rw_cond_params.items()}})

    # Choice Kernel model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    beta_bound = (40, 200)
    ck_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'beta': np.random.uniform(*beta_bound)
    }
    choice_kernel_sim_conf = CK_sim((ck_params['alpha'], ck_params['sigma'], ck_params['beta']), confidence, n_trials)
    participant_results.update({'ck_conf': choice_kernel_sim_conf, **{f'ck_{k}': v for k, v in ck_params.items()}})

    # RW + Choice Kernel model parameters and simulation
    alpha_bound = (0, 1)
    alpha_ck_bound = (0, 1)
    sigma_bound = (1, 10)
    sigma_ck_bound = (1, 10)
    beta_bound = (40, 200)
    beta_ck_bound = (40, 200)
    rwck_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'alpha_ck': np.random.uniform(*alpha_ck_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'sigma_ck': np.random.uniform(*sigma_ck_bound),
        'beta': np.random.uniform(*beta_bound),
        'beta_ck': np.random.uniform(*beta_ck_bound)
    }
    rwck_sim_conf = RWCK_sim((rwck_params['alpha'], rwck_params['alpha_ck'], rwck_params['sigma'], rwck_params['sigma_ck'], rwck_params['beta'],  rwck_params['beta_ck']), feedback, confidence, n_trials)
    participant_results.update({'rwck_conf': rwck_sim_conf, **{f'rwck_{k}': v for k, v in rwck_params.items()}})

    # Rescorla-Wagner Performance Delta model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    w_rw_bound = (0, 1)
    w_delta_p_bound = (1, 3)
    intercept_bound = (0, 100)

    a, b = 2, 10
    low, high = w_rw_bound
    low, high = intercept_bound
    intercept_beta = low + beta.rvs(a, b) * (high - low)

    rwpd_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'w_rw':  np.random.uniform(*w_rw_bound),
        'w_delta_p': np.random.uniform(*w_delta_p_bound),
        'intercept': intercept_beta,  # now sampled from a right-skewed beta
    }

    rwpd_sim_conf = RWPD_sim((rwpd_params['alpha'], rwpd_params['sigma'], rwpd_params['w_rw'], rwpd_params['w_delta_p'], rwpd_params['intercept']), confidence, feedback, n_trials, performance)
    participant_results.update({'rwpd_conf': rwpd_sim_conf, **{f'rwpd_{k}': v for k, v in rwpd_params.items()}})

    # Rescorla-Wagner Performance model parameters and simulation
    alpha_bound = (0, 1)
    sigma_bound = (1, 10)
    w_rw_bound = (0, 1)
    w_p_bound = (0.2, 1)
    intercept_bound = (0, 50)

    # Beta distribution parameters for right-skew
    a, b = 2, 10
    low, high = w_rw_bound
    w_rw_beta = low + beta.rvs(a, b) * (high - low)
    low, high = w_p_bound
    w_p_beta = low + beta.rvs(a, b) * (high - low)
    low, high = intercept_bound
    intercept_beta = low + beta.rvs(a, b) * (high - low)

    rwp_params = {
        'alpha': np.random.uniform(*alpha_bound),
        'sigma': np.random.uniform(*sigma_bound),
        'w_rw': w_rw_beta, # np.random.uniform(*w_rw_bound),
        'w_p': np.random.uniform(*w_p_bound),
        'intercept': intercept_beta,  # now sampled from a right-skewed beta
    }

    rwp_sim_conf, y_val_rwp = RWP_sim((rwp_params['alpha'], rwp_params['sigma'], rwp_params['w_rw'], rwp_params['w_p'], rwp_params['intercept']), confidence, feedback, n_trials, performance)
    participant_results.update({'rwp_conf': rwp_sim_conf, **{f'rwp_{k}': v for k, v in rwp_params.items()}})

    return [participant_results, participant, session]

def main(df):
    # Get unique pid and session pairs
    unique_pairs = df[['pid', 'session']].drop_duplicates()
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))

    # Create a list of DataFrames, one for each unique pid-session pair
    df_list = [df[(df['pid'] == pid) & (df['session'] == session)].copy()
               for pid, session in unique_pairs_list]

    # Define model names for easier labeling in the DataFrame
    model_names = [
        "bias", "wsls", "rw", "rw_cond",
        "ck", "rwck", "rwpd", "rwp"
    ]

    # List to collect trial data and model parameters for the final DataFrame
    all_trial_data = []

    # Run the process_session function for each session
    # (i.e., one run for each participant)
    for session_df in tqdm(df_list, total=len(df_list),
                           desc='Simulating models'):
        # Call process_session and unpack results, participant, and session
        session_results, participant_id, session_id = process_session(session_df)

        # Iterate over each model's results and parameters
        for model_name in model_names:
            # Access confidence data for this model dynamically
            confidence_key = f'{model_name}_conf'
            confidence_data = session_results.get(confidence_key)

            if confidence_data is None:
                print(f"Warning: No confidence data found for model \
                      {model_name} in session results.")
                continue

            # Extract parameters for the current model by filtering out '_conf' keys
            parameters = {k: v for k, v in session_results.items()
                          if k.startswith(model_name)
                          and not k.endswith('_conf')}

            # Create trial data with confidence and parameters
            for trial_idx, confidence in enumerate(confidence_data):
                trial_data = {
                    'confidence_sim': confidence,
                    'model': model_name,
                    'trial': trial_idx + 1,
                    'session': session_id,
                    'pid': participant_id,
                    **parameters
                }
                all_trial_data.append(trial_data)

    # Convert the trial data into a DataFrame
    results_df = pd.DataFrame(all_trial_data)

    return results_df

def plot_model_results_with_real_data(df, df_real):
    """
    Plot results for each model with three columns: real data, individual session outcomes,
    and mean confidence over time.

    Parameters:
    - df (pd.DataFrame): DataFrame with columns ['confidence_sim', 'model', 'trial', 'session', 'pid'].
    - df_real (pd.DataFrame): DataFrame with columns ['confidence_real', 'model', 'trial', 'session', 'pid'].
    """
    # Get unique model names to determine the number of rows
    model_names = df['model'].unique()
    n_models = len(model_names)

    # Set up a 3-column subplot structure (real data, individual simulations, mean confidence)
    fig, axes = plt.subplots(n_models, 3, figsize=(20, 4 * n_models),
                             sharex='col', sharey='row')
    if n_models == 1:
        axes = [axes]  # Ensure axes is always a list for consistency

    # Plot each model as separate rows
    for model_idx, model_name in enumerate(model_names):
        # Separate the three subplots for this model
        ax_real = axes[model_idx][0]  # Real data plot
        ax_individual = axes[model_idx][1]  # Individual simulations
        ax_mean = axes[model_idx][2]  # Mean confidence

        # Filter data for the current model
        model_df = df[df['model'] == model_name]
        model_df_real = df_real.copy()

        # Plot real data in the first column
        for session_id in model_df_real['session'].unique():
            session_df_real = model_df_real[model_df_real['session'] == session_id]
            ax_real.plot(session_df_real['trial'],
                         session_df_real['confidence'],
                         label=f'Session {session_id}', marker='o',
                         color='C0', alpha=0.7)

        # Plot each session's confidence for the current model in the second column (simulations)
        for session_id in model_df['session'].unique():
            session_df = model_df[model_df['session'] == session_id]
            ax_individual.plot(session_df['trial'],
                               session_df['confidence_sim'],
                               label=f'Session {session_id}', marker='o', alpha=0.7)

        # Calculate and plot the mean confidence over trials for the current model in the third column
        mean_confidence = model_df.groupby('trial')['confidence_sim'].mean()
        mean_confidence_real = model_df_real.groupby('trial')['confidence'].mean()
        ax_mean.plot(mean_confidence_real.index, mean_confidence_real.values, color='black',
                     marker='o', linestyle='-', linewidth=2, label="Real")
        ax_mean.plot(mean_confidence.index, mean_confidence.values, color='b',
                     marker='o', linestyle='-', linewidth=2, label="Simulated")

        # Set titles and labels
        ax_real.set_title(f'Real Data', fontsize=25)
        ax_individual.set_title(f'{model_name}', fontsize=25)
        ax_mean.set_title(f'Mean Confidence Over Trials', fontsize=25)

        ax_real.set_ylabel('Confidence', fontsize=25)
        ax_real.spines[['top', 'right']].set_visible(False)
        ax_individual.spines[['top', 'right']].set_visible(False)
        ax_mean.spines[['top', 'right']].set_visible(False)

        # Set y-axis limits
        ax_real.set_ylim(-5, 105)
        ax_individual.set_ylim(-5, 105)
        ax_mean.set_ylim(-5, 105)

        # Set y-ticks
        for ax in [ax_real, ax_individual, ax_mean]:
            ax.set_yticks(range(0, 101, 20))
            ax.set_yticklabels(range(0, 101, 20), fontsize=25)

    # Label the shared x-axis
    axes[-1][0].set_xlabel('Trial', fontsize=25)
    axes[-1][1].set_xlabel('Trial', fontsize=25)
    axes[-1][2].set_xlabel('Trial', fontsize=25)

    plt.legend()
    plt.tight_layout()

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path
    relative_path = "../../results/variable_feedback/model_comparison/model_and_param_recovery"

    # Construct the full path to the CSV file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))
    file_name = r"model_simulations_EXP2"
    save_path = os.path.join(relative_path, file_path, file_name)

    plt.savefig(f'{save_path}.png', dpi=300)

    plt.show()

def plot_estimates_by_pid(df):
    """
    Plots a line plot for each participant (pid) with 'estimate' on the Y-axis
    and 'correct' on the X-axis.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing 'pid', 'estimate', and 'correct' columns.
    """

    # Get unique participant IDs
    unique_pids = df['pid'].unique()

    # create error column
    df['abs_error'] = df.estimate-df.correct

    # Set up the plot
    plt.figure(figsize=(5,5))

    # Plot each participant's data
    for pid in unique_pids:
        pid_df = df[df['pid'] == pid]
        plt.scatter(pid_df['correct'], pid_df['estimate'], marker='o', linestyle='-', label=f'PID {pid}', alpha=0.7)

    # Formatting
    plt.xlabel("Correct", fontsize=12)
    plt.ylabel("Estimate", fontsize=12)
    plt.title("Estimates vs. Correct by Participant", fontsize=14)
    #plt.legend(title="Participant ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True)
    plt.axis('equal')

    # Truth
    plt.plot(pid_df['correct'], pid_df['correct'], marker='o', linestyle='-',  alpha=1, color='k')

    plt.show()


def plot_performance_on_confidence_by_pid(df):
    """

    """

    # Get unique participant IDs
    unique_pids = df['pid'].unique()

    # create error column
    df['abs_error'] = df.estimate-df.correct

    # Set up the plot
    plt.figure(figsize=(5,5))

    # Plot each participant's data
    for pid in unique_pids:
        pid_df = df[df['pid'] == pid]
        plt.scatter(pid_df['abs_error'], pid_df['confidence'], marker='o', linestyle='-', label=f'PID {pid}', alpha=0.7)

    # Formatting
    plt.xlabel("abs_error", fontsize=12)
    plt.ylabel("confidence", fontsize=12)
    plt.title("abs_error vs. confidence by Participant", fontsize=14)
    plt.grid(True)

    # Truth
    #plt.plot(pid_df['correct'], pid_df['correct'], marker='o', linestyle='-',  alpha=1, color='k')

    plt.show()

#-----------------------------------------------------------------------------
# Execution
#-----------------------------------------------------------------------------
if __name__ == '__main__':

    # Import data - Variable feedback condition (Experiment 2)
    df = load_df(EXP=2)

    # Add session column
    df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

    # Plots or debugging
    # plot_estimates_by_pid(df) # Dot estimate
    # plot_performance_on_confidence_by_pid(df) # Abs error vs confidence

    # Run main
    results_df = main(df)

    # plot
    plot_model_results_with_real_data(results_df, df)

    # Merge results_df into df based on 'trial', 'participant', and 'session'
    merged_df = df.merge(results_df, on=['trial', 'pid', 'session'],
                         how='left')

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Relative path
    relative_path = "../../results/variable_feedback/model_comparison/model_and_param_recovery"

    # Construct the full path to the CSV file
    file_path = os.path.normpath(os.path.join(script_dir, relative_path))
    file_name = r"model_simulations_EXP2"
    save_path = os.path.join(relative_path, file_path, file_name)

    # Save simulated data
    merged_df.to_csv(f'{save_path}.csv', index=False)



