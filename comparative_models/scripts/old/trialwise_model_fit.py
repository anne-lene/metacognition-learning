# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:01:56 2023

@author: carll
"""

# Model fit trial by trial - fixed feedback

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
from matplotlib.image import imread
from tqdm import tqdm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from src.utility_functions import (create_truncated_normal,
                                   add_session_column,
                                   HandlerNormalDist)

from src.models import (fit_model,
                        random_model_w_bias,
                        random_model_w_bias_trial,
                        win_stay_lose_shift,
                        win_stay_lose_shift_trial,
                        rw_symmetric_LR,
                        rw_symmetric_LR_trial,
                        rw_cond_LR,
                        rw_cond_LR_trial,
                        choice_kernel,
                        choice_kernel_trial,
                        RW_choice_kernel,
                        RW_choice_kernel_trial,
                        delta_P_RW,
                        delta_P_RW_trial
                        )

# Import data - fixed feedback condition
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_path = r"fixed_feedback"
fixed_feedback_data_path = r'data/cleaned'
data_file = r'main-20-12-14-processed_filtered.csv'
full_path = os.path.join(project_path, experiment_path,
                         fixed_feedback_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

# Add session column
df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Create empty plot to store model prediction info
complete_df = pd.DataFrame({})

# Loop over participants
for pid_nr, participant in enumerate(tqdm(df.pid.unique(),
                                          total=len(df.pid.unique()),
                                          desc='Participant loop')):

    df_p = df[df.pid==participant]

    # Loop over sessions
    for session in df_p.session.unique():

# =============================================================================
#         # Get current session data, one row per trial
#         df_s = df_p[df_p.session == session]
#         df_s = df_s.drop_duplicates(subset='trial', keep='first')
#
#         # Identify the last row of the 'baseline' condition
#         last_baseline_row = df_s[df_s['condition'] == 'baseline'].iloc[-1]
#
#         # Filter out all 'baseline' rows
#         df_s = df_s[df_s['condition'] != 'baseline']
#
#         # Prepend the last baseline row to the start of df_s
#         df_s = pd.concat([last_baseline_row.to_frame().T, df_s],
#                          ignore_index=True)
# =============================================================================

        # Get current session data, one row per trial
        df_s = df_p[df_p.session == session]

        # Only feedback trials
        df_s = df_s[df_s.condition != 'baseline']

        # Calculate the difference between 'estimate' and 'correct'
        df_s['difference'] = df_s['estimate'] - df_s['correct']

        # Group by 'subtrial' and calculate the mean of the differences
        error_avg = df_s.groupby('trial')['difference'].mean()

        # One row per trial
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        n_trials = len(df_s)
        confidence = df_s.confidence.values
        feedback = df_s.feedback.values
        condition = df_s.condition.values
        performance =  -error_avg.values #df_s.estimate.values - df_s.correct.values #df_s.pavg.values

        # Bias model
        # Set Bounds
        mean_bound = (1, 100)  # Standard deviation
        sigma_bound = (1, 100)    # Win boundary
        bounds = [(mean_bound[0],  mean_bound[1]),
                  (sigma_bound[0],  sigma_bound[1])]

        # Ensure confidence is not above 100, if so truncate to 2 first digits
        confidence = np.array([int(c // 10**(len(str(c)) - 2))
                                     if c >= 100
                                     else int(c)
                                     for c in confidence])

        # Get results
        results_bias = fit_model(model=random_model_w_bias,
                                     args=(confidence,
                                           n_trials),
                                     bounds=bounds,
                                     n_trials=n_trials,
                                     start_value_number=10,
                                     solver="L-BFGS-B")

        best_mean_bias = results_bias[0]
        best_std_bias = results_bias[1]
        nll_bias = results_bias[2]
        aic_bias = results_bias[3]
        bic_bias = results_bias[4]


        trial_results_bias = random_model_w_bias_trial((best_mean_bias,
                                                        best_std_bias),
                                              confidence,
                                              n_trials)

        nlls_bias = trial_results_bias[0]
        prob_dists_bias = trial_results_bias[1]
        confidence_clean_bias = trial_results_bias[2]

        # WSLS model
        # Set Bounds
        sigma_bound = (1, 100)  # Standard deviation
        win_bound = (1, 100)    # Win boundary
        bounds = [(sigma_bound[0],  sigma_bound[1]),
                  (win_bound[0],  win_bound[1])]

        # Get results
        results_win_stay = fit_model(model=win_stay_lose_shift,
                                     args=(confidence,
                                           feedback,
                                           n_trials
                                           ),
                                     bounds=bounds,
                                     n_trials=n_trials,
                                     start_value_number=10,
                                     solver="L-BFGS-B")

        best_std_WSLS = results_win_stay[0]
        best_win_boundary_WSLS = results_win_stay[1]
        nll_WSLS = results_win_stay[2]
        aic_WSLS = results_win_stay[3]
        bic_WSLS = results_win_stay[4]

        # Trial-wise NLL and probability distribution
        trial_results_WSLS = win_stay_lose_shift_trial(
                                                (best_std_WSLS,
                                                 best_win_boundary_WSLS),
                                                confidence,
                                                feedback,
                                                n_trials)

        nlls_WSLS = trial_results_WSLS[0]
        prob_dists_WSLS = trial_results_WSLS[1]
        confidence_clean = trial_results_WSLS[2]


        # RW model
        # Set bounds
        alpha_bound = (0.001, 1)  # Alpha
        sigma_bound = (1, 100)    # Standard deviation
        bias_bound = (0, 100)     # Mean at first trial
        bounds = [(alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1])]

        # Get results
        results_rw_symm = fit_model(model=rw_symmetric_LR,
                                    args=(confidence,
                                          feedback,
                                          n_trials),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=10,
                                    solver="L-BFGS-B")

        best_alpha = results_rw_symm[0]
        best_std = results_rw_symm[1]
        best_bias = results_rw_symm[2]
        nll = results_rw_symm[3]
        aic = results_rw_symm[4]
        bic = results_rw_symm[5]
        pseudo_r2 = results_rw_symm[6]

        trial_results = rw_symmetric_LR_trial((best_alpha,
                                               best_std,
                                               best_bias),
                                              confidence,
                                              feedback,
                                              n_trials)

        nlls_trial = trial_results[0]
        model_pred = trial_results[1]
        sigma_vec = trial_results[2]
        confidence_clean = trial_results[3]


        # RW Cond model
        alpha_neut_bound = (0.001, 1)  # Alpha neut
        alpha_pos_bound = (0.001, 1)  # Alpha pos
        alpha_neg_bound = (0.001, 1)  # Alpha neg
        sigma_bound = (1, 100)    # Standard deviation
        bias_bound = (0, 100)     # Mean at first trial
        bounds = [(alpha_neut_bound[0], alpha_neut_bound[1]),
                  (alpha_pos_bound[0], alpha_pos_bound[1]),
                  (alpha_neg_bound[0], alpha_neg_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1])]

        # Get results
        results_rw_cond = fit_model(model=rw_cond_LR,
                                    args=(confidence,
                                          feedback,
                                          n_trials,
                                          condition),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=10,
                                    solver="L-BFGS-B")

        best_alpha_neut = results_rw_cond[0]
        best_alpha_pos = results_rw_cond[1]
        best_alpha_neg = results_rw_cond[2]
        best_std = results_rw_cond[3]
        best_bias = results_rw_cond[4]
        nll = results_rw_cond[5]
        aic = results_rw_cond[6]
        bic = results_rw_cond[7]
        pseudo_r2 = results_rw_cond[8]

        rw_cond_trial_results = rw_cond_LR_trial((best_alpha_neut,
                                                  best_alpha_pos,
                                                  best_alpha_neg,
                                                  best_std,
                                                  best_bias),
                                                 confidence,
                                                 feedback,
                                                 n_trials,
                                                 condition)

        nlls_trial_rw_cond = rw_cond_trial_results[0]
        model_pred_rw_cond = rw_cond_trial_results[1]
        sigma_vec_rw_cond = rw_cond_trial_results[2]
        confidence_clean_rw_cond = rw_cond_trial_results[3]



        # Choice Kernel Model

        # Set bounds
        alpha_bound = (0.001, 1)  # Alpha
        sigma_bound = (1, 100)    # Standard deviation
        bias_bound = (0, 100)     # Mean at first trial
        beta_bound = (0, 5)        # Beta
        bounds = [
                  (alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1]),
                  (beta_bound[0], beta_bound[1])
                  ]

        # Get results
        results_ck = fit_model(model=choice_kernel,
                                    args=(confidence,
                                          n_trials,
                                          ),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=10,
                                    solver="L-BFGS-B")

        best_alpha = results_ck[0]
        best_std = results_ck[1]
        best_bias = results_ck[2]
        best_beta = results_ck[3]
        nll = results_ck[4]
        aic = results_ck[5]
        bic = results_ck[6]
        pseudo_r2 = results_ck[7]

        choice_kernel_trial_results = choice_kernel_trial((best_alpha,
                                                           best_std,
                                                           best_bias,
                                                           best_beta),
                                                          confidence,
                                                          n_trials)

        nlls_trial_choice_kernel = choice_kernel_trial_results[0]
        model_prob_choice_kernel = choice_kernel_trial_results[1]
        sigma_vec_choice_kernel = choice_kernel_trial_results[2]
        confidence_clean_choice_kernel = choice_kernel_trial_results[3]


        # RW + Choice Kernel Model

        # Set bounds
        alpha_bound = (0.001, 1)  # Alpha
        alpha_ck_bound = (0.001, 1)  # Alpha choice kernel
        sigma_bound = (1, 100)    # Standard deviation
        bias_bound = (0, 100)     # Mean at first trial
        beta_bound = (0, 5)        # Beta
        bounds = [
                  (alpha_bound[0], alpha_bound[1]),
                  (alpha_ck_bound[0], alpha_ck_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1]),
                  (beta_bound[0], beta_bound[1])
                  ]

        # Get results
        results_rwck = fit_model(model=RW_choice_kernel,
                                    args=(feedback,
                                          confidence,
                                          n_trials,
                                          ),
                                    bounds=bounds,
                                    n_trials=n_trials,
                                    start_value_number=10,
                                    solver="L-BFGS-B")

        best_alpha = results_rwck[0]
        best_alpha_ck = results_rwck[1]
        best_std = results_rwck[2]
        best_bias = results_rwck[3]
        best_beta = results_rwck[4]
        nll = results_rwck[5]
        aic = results_rwck[6]
        bic = results_rwck[7]
        pseudo_r2 = results_rwck[8]

        RW_choice_kernel_trial_results = RW_choice_kernel_trial((best_alpha,
                                                                 best_alpha_ck,
                                                                 best_std,
                                                                 best_bias,
                                                                 best_beta),
                                                                feedback,
                                                                confidence,
                                                                n_trials)

        nlls_trial_RW_choice_kernel = RW_choice_kernel_trial_results[0]
        model_prob_RW_choice_kernel = RW_choice_kernel_trial_results[1]
        sigma_vec_RW_choice_kernel = RW_choice_kernel_trial_results[2]
        confidence_clean_RW_choice_kernel = RW_choice_kernel_trial_results[3]


        # Delta P + Rescorla wagner model

        # Set bounds
        alpha_bound = (0.001, 1)    # Alpha
        sigma_bound = (1, 100)      # Standard deviation
        bias_bound = (0, 100)       # Mean at first trial
        w_rw_bound = (-1, 1)       # Weight for RW update
        w_delta_p_bound = (-1, 1)  # Weight for delta p
        bounds = [(alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1]),
                  (bias_bound[0], bias_bound[1]),
                  (w_rw_bound[0], w_rw_bound[1]),
                  (w_delta_p_bound[0], w_delta_p_bound[1])]

        # Get results
        results_delta_p_rw = fit_model(model=delta_P_RW,
                                       args=(confidence,
                                              feedback,
                                              n_trials,
                                              performance),
                                       bounds=bounds,
                                       n_trials=n_trials,
                                       start_value_number=10,
                                       solver="L-BFGS-B")

        best_alpha = results_delta_p_rw[0]
        best_std = results_delta_p_rw[1]
        best_bias = results_delta_p_rw[2]
        best_w_rw = results_delta_p_rw[3]
        best_w_delta_p = results_delta_p_rw[4]
        nll = results_delta_p_rw[5]
        aic = results_delta_p_rw[6]
        bic = results_delta_p_rw[7]
        pseudo_r2 = results_delta_p_rw[8]

        delta_P_RW_trial_results = delta_P_RW_trial((best_alpha,
                                                     best_std,
                                                     best_bias,
                                                     best_w_rw,
                                                     best_w_delta_p),
                                                    confidence,
                                                    feedback,
                                                    n_trials,
                                                    performance)

        nlls_trial_delta_P_RW = delta_P_RW_trial_results[0]
        model_pred_delta_P_RW = delta_P_RW_trial_results[1]
        sigma_vec_delta_P_RW = delta_P_RW_trial_results[2]
        confidence_clean_delta_P_RW = delta_P_RW_trial_results[3]

        # GET MODEL PREDICTION DATA  ------------------------------------------
        # Set plot variables
        n_trials = 20 # task trials
        n_options = 101

        models = []
        trials = []
        probabilities = []
        options = []

        pid = []
        condition = []
        session = []

        # Start from second trial as only these were fit
        for trial in range(n_trials-1):

            for option in range(0, n_options):
                # Random model
                models.append('Random')
                trials.append(trial)
                probabilities.append(1/n_options)
                options.append(option)

                # Bias model
                models.append('Bias')
                trials.append(trial)
                probabilities.append(prob_dists_bias[trial][option])
                options.append(option)

                # WSLS model
                models.append('WSLS')
                trials.append(trial)
                probabilities.append(prob_dists_WSLS[trial][option])
                options.append(option)

                # RW model
                dist = create_truncated_normal(model_pred[trial],
                                               sigma_vec[trial],
                                               0,
                                               n_options)

                models.append('RW')
                trials.append(trial)
                probabilities.append(dist.pdf(option))
                options.append(option)

                # RW_cond model
                if experiment_path != 'fixed_feedback':

                    dist = create_truncated_normal(model_pred_rw_cond[trial],
                                                   sigma_vec_rw_cond[trial],
                                                   0,
                                                   n_options)

                    models.append('RW_cond')
                    trials.append(trial)
                    probabilities.append(dist.pdf(option))
                    options.append(option)

                # Choice Kernel model
                models.append('Choice_Kernel')
                trials.append(trial)
                probabilities.append(model_prob_choice_kernel[trial][option])
                options.append(option)

                # Choice RW + Kernel model
                models.append('RW_Choice_Kernel')
                trials.append(trial)
                probabilities.append(model_prob_RW_choice_kernel[trial][option])
                options.append(option)

                # Delta P + RW
                dist = create_truncated_normal(model_pred_delta_P_RW[trial],
                                               sigma_vec_delta_P_RW[trial],
                                               0,
                                               n_options)
                models.append('Delta_P_RW')
                trials.append(trial)
                probabilities.append(dist.pdf(option))
                options.append(option)

        # Create DataFrame
        df_plot = pd.DataFrame({'model': models,
                                'trial': trials,
                                'probability': probabilities,
                                'option': options,
                                'trial_count': [i+1 for i in trials],
                                'session': [session for i in
                                            range(len(trials))],
                                'condition': [df_s.condition.unique()[0]
                                              for i in range(len(trials))],
                                'pid': [pid_nr for i in range(len(trials))]
                                })

        # Concat df_plot to complete df with all model info across sessions
        complete_df = pd.concat([complete_df, df_plot], ignore_index=True)

        # PLOT INDIVIDUAL MODEL PREDICTIONS -----------------------------------
        # Function to plot with a line and filled area
        def lineplot_with_fill(*args, **kwargs):
            data = kwargs.pop('data')
            ax = plt.gca()
            x = data['option']
            y = data['probability']
            ax.plot(x, y, **kwargs)
            ax.fill_between(x, y, color=kwargs.get('color', 'blue'), alpha=1)
            ax.set_facecolor('none')

        # Function to add a vertical line
        def add_vertical_line(x, **kwargs):
            ax = plt.gca()
            ax.axvline(x=x, **kwargs)

        # Set color pallet
        pal = sns.cubehelix_palette(4, start=0, rot=-.25, light=.7)

        # Iterate over each model to create separate plots
        model_list = df_plot.model.unique()
        for c, model_name in enumerate(model_list):

            # Filter the DataFrame for the current model
            model_df = df_plot[df_plot['model'] == model_name]

            # Initialize the FacetGrid object for the current model
            g = sns.FacetGrid(model_df, row="trial", hue="trial",
                              aspect=10, height=.5, palette=pal)

            # Draw the lineplot with fill
            g.map_dataframe(lineplot_with_fill, 'option', 'probability')

            # Plot the vertical line for each trial
            for trial, ax in g.axes_dict.items():
                confidence_value = confidence_clean_bias[trial]
                ax.axvline(x=confidence_value, color='red', linestyle='-')

            # Adjust the layout
            g.fig.subplots_adjust(hspace=-.55)

            # Remove axes details that don't play well with overlap
            g.set_titles("")
            g.set(yticks=[], ylabel='')
            g.despine(bottom=True, left=True)

            # Turn off the gridlines
            g.set(xticks=[], yticks=[])
            g.despine(left=True, bottom=True)

            # Add a number to the left of each row
            for ax, (row_val, _) in zip(g.axes.flatten(),
                                        model_df.groupby("trial")):
                # Get trial number
                trial_number = row_val+1
                ax.text(-0.05, 0.5, trial_number,
                        transform=ax.transAxes,
                        ha="right", va="center",
                        fontsize=19)
                model_name_ = f'{model_name}'
                if row_val == 0:
                    ax.text(0.5, 1.4, model_name_,
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=19)

            # Set fontsize for x-tick labels
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_size(19)
                ax.set_xlabel('Confidence', fontsize=19)



            # Save and Show the plot
            plt.savefig(f'{model_name}_plot.png',
                        bbox_inches='tight',
                        dpi=300)
            plt.show()


        #% PLOT SESSION SUMMARY ---------------------------------------------
        # create full figure
        fig = plt.figure(figsize=(24, 8))

        # Define the grid layout
        gs = fig.add_gridspec(9, 7)  # 5 rows, 4 columns

        # Create subplots
        ax_images = [fig.add_subplot(gs[0:len(model_list), i])
                     for i in range(len(model_list))]

        # Loop through the axes and image paths and set img to axes
        image_paths = [f"{i}_plot.png" for i in model_list]

# =============================================================================
#         image_paths = ['Random_plot.png', 'Bias_plot.png',
#                        'WSLS_plot.png', 'RW_plot.png',
#                        f'{model_list[4]}.png', f'{model_list[4]}.png',
#                        f'{model_list[4]}.png', f'{model_list[4]}.png']
# =============================================================================
        for ax_nr, (ax, img_path) in enumerate(zip(ax_images, image_paths)):
            img = imread(img_path)
            ax.imshow(img)
            if ax_nr == 0:
                # Hide the x-axis labels and ticks
                ax.set_xticks([])
                ax.set_xticklabels([])

                # Hide the y-axis ticks (but keep the labels)
                ax.set_yticks([])
                ax.set_ylabel('Trial')
                ax.spines[['top',
                           'right',
                           'left',
                           'bottom']].set_visible(False)
            else:
                ax.axis('off')  # Hide the axes ticks and labels
            if ax_nr == len(ax_images)-1:
                red_line = mlines.Line2D([], [], color='red', marker='|',
                                         markersize=15,
                                         label='Participant Choice',
                                         linestyle='None')

                # Distribution symbol patch
                dummy_line = mpatches.Patch(
                    color='steelblue',
                    label='Model Prediction Probability')

                ax.legend([red_line, dummy_line],
                          ['Participant Choice',
                           'Model Prediction\nProbability'],
                          handler_map={mpatches.Patch: HandlerNormalDist()},
                          bbox_to_anchor=(1.8, 1),
                          facecolor='white',
                          framealpha=1)


        # NLL plot
        ax = fig.add_subplot(gs[7:, :])     # Span fifth row, all columns
        x = range(1, n_trials)

        # Random model
        nlls_trail_random = np.full(19, -np.log(1/100))
        ax.plot(x, nlls_trail_random, c='blue', label='Random')

        # Bias model
        ax.plot(x, nlls_bias, c='green', label='Biased')

        # WSLS
        ax.plot(x, nlls_WSLS, c='r', label='WSLS')

        # RW
        ax.plot(x, nlls_trial, c='purple', label='RW')

        # RW_Cond
        if experiment_path != 'fixed_feedback':
            ax.plot(x, nlls_trial_rw_cond, c='cyan', label='RW_Cond')

        # Choice Kernel
        ax.plot(x, nlls_trial_choice_kernel, c='pink', label='CK')

        # RW + Choice Kernel
        ax.plot(x, nlls_trial_RW_choice_kernel, c='orange', label='RWCK')

        # RW + Choice Kernel
        ax.plot(x, nlls_trial_delta_P_RW, c='silver', label='RWPD')

        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        ax.set_xlabel('Trial')
        ax.set_ylabel('NLL')

        # Set condition as title
        condition = df_s.condition.unique()[0]
        if condition == 'neut':
            condition_string = "Neutral Feedback"
        if condition == 'pos':
            condition_string = "Positive Feedback"
        if condition == 'neg':
            condition_string = "Negative Feedback"
        plt.suptitle(f'{condition_string} | Participant {pid_nr}')

        ax.legend(bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)

        #plt.tight_layout()

        # Set save path
        if experiment_path == 'fixed_feedback':
            result_path = r"results\Fixed_feedback\Trial_by_trial_fit"
        else:
            result_path = r"results\variable_feedback\trial_by_trial_fit"
        file_name = f'{pid_nr}_{participant}_{condition}_trial_plot.png'
        save_path = os.path.join(project_path, 'comparative_models',
                                 result_path, file_name)

        # Save
        plt.savefig(save_path,
                    bbox_inches='tight',
                    dpi=300)
        # Show plot
        plt.show()

#%% Save model trial data

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_fit"
else:
    result_path = r"results\variable_feedback\trial_by_trial_fit"

filename = 'model_trial_data.csv'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, filename)

# Save the DataFrame to an Excel file
complete_df.to_csv(save_path, index=False)


#%% Plot the average model max probabilities

def get_probability_dict(df):
    # Dictionary to store the results with keys as (pid, condition) and values as arrays of probabilities
    results = {}

    # Iterate over each combination of 'pid' and 'condition'
    for (pid, condition), group in df.groupby(['pid', 'condition']):

        # Sort the group by 'trial' and then 'option' to ensure consistent ordering
        group_sorted = group.sort_values(by=['trial', 'option'])

        # Find the unique number of trials and options to reshape the array accordingly
        n_trials = group_sorted['trial'].nunique()
        n_options = group_sorted['option'].nunique()

        # Reshape the probabilities into a 2D array where each row is a trial
        probabilities = group_sorted['probability'].values.reshape(n_trials, n_options)

        # Store the result in the dictionary
        results[(pid, condition)] = probabilities

    return results

def random_argmax(arr):
    # Step 1: Find the maximum value in the array
    max_value = np.max(arr)

    # Step 2: Find all indices where the array equals the maximum value
    max_indices = np.flatnonzero(arr == max_value)

    # Step 3: Randomly choose one index from the max indices
    chosen_index = np.random.choice(max_indices)

    return chosen_index


fig, [[ax,ax2,ax3,ax4], [ax5,ax6,ax7,ax8]] = plt.subplots(2,4,
                                                          figsize=(11,4),
                                                          sharex=True,
                                                          sharey=True)

# Check experiment (varied or fixed feedback)
if experiment_path != 'fixed_feedback':
    colors = ['blue', 'green', 'red', 'purple',
              'cyan', 'pink', 'orange', 'silver']
else:
    colors = ['blue', 'green', 'red', 'purple', 'pink', 'orange', 'silver']

# Loop over models
for model, color, axis in tqdm(zip(complete_df.model.unique(), colors,
                             [ax, ax2, ax3, ax4, ax5, ax6, ax7, ax8]),
                         total=len(colors)):

    # Filter on current model
    df_model = complete_df[complete_df.model==model]

    # Get probability dictionary (contains one prob array per session)
    prob_dict = get_probability_dict(df_model)

    # Get the maximum probability option
    pid = []
    condition = []
    prob_max_idx_list = []
    for key in prob_dict.keys():
        prob_array = prob_dict[key[0], key[1]] # Session array
        # If two or more values are highest, choose one of them at random
        prob_max_idx = [random_argmax(row) for row in prob_array]
        pid.append(key[0])
        condition.append(key[1])
        prob_max_idx_list.append(prob_max_idx)

    # Put prob_max_idx into dataframe
    d = {'pid': pid, 'condition': condition, 'prob_max_idx':prob_max_idx_list}
    df_model_max_idx = pd.DataFrame(d)

    # Make entries arrays
    df_model_max_idx['prob_max_idx'] = df_model_max_idx['prob_max_idx'].apply(np.array)


    # Plot
    for cond, color in zip(['neut', 'neg', 'pos'],
                           ['grey', 'red', 'green']):

        # Filter on condition
        df_plot = df_model_max_idx[df_model_max_idx.condition==cond]

        # Calculate mean and sem across columns
        means = np.mean(df_plot.prob_max_idx.values, axis=0)
        std = np.std(df_plot.prob_max_idx.values, axis=0, ddof=1)
        sem = std/np.sqrt(len(df_plot.prob_max_idx.values))

        # Plot
        x = range(1, len(means)+1)
        axis.plot(x, means, color=color)
        axis.fill_between(x, means+sem, means-sem,
                          color=color, alpha=0.2)


    axis.set_ylim(38,56)
    axis.set_xticks(range(1, len(means)+1, 2))
    axis.set_xticklabels(range(2, len(means)+2, 2))
    axis.set_xlabel('Trials')
    axis.set_ylabel('Confidence')
    if model == 'Delta_P_RW':
        axis.set_title('RW + Performance Delta')
    elif model == 'Choice_Kernel':
        axis.set_title('Choice Kernel')
    elif model == 'RW_Choice_Kernel':
        axis.set_title('RW + Choice Kernel')
    else:
        axis.set_title(model)

    axis.spines[['top', 'right']].set_visible(False)

    # Hide empty subplot
    if experiment_path == r"fixed_feedback":
        ax8.set_visible(False)
    else:
        pass

plt.tight_layout()
plt.subplots_adjust(hspace=0.7)

# Set save path
if experiment_path == 'fixed_feedback':
    result_path = r"results\Fixed_feedback\Trial_by_trial_fit"
else:
    result_path = r"results\variable_feedback\trial_by_trial_fit"

filename = 'model_max_prob_averge.svg'
save_path = os.path.join(project_path, 'comparative_models',
                         result_path, filename)

# Save
#plt.savefig(save_path,
#            bbox_inches='tight',
#            dpi=300)

plt.show()






