# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:01:56 2023

@author: carll
"""

# Model fit trial by trial

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import seaborn as sns
from matplotlib.image import imread
from tqdm import tqdm
import matplotlib.lines as mlines
from src.utility_functions import (create_truncated_normal,
                                   add_session_column)
from src.models import (fit_model,
                        random_model_w_bias,
                        random_model_w_bias_trial,
                        win_stay_lose_shift,
                        win_stay_lose_shift_trial,
                        rw_symmetric_LR,
                        rw_symmetric_LR_trial,
                        )



# Import data
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

# Loop over participants
for pid_nr, participant in enumerate(tqdm(df.pid.unique(),
                                          total=len(df.pid.unique()),
                                          desc='Participant loop')):

    df_p = df[df.pid==participant]

    # Loop over sessions
    for session in df_p.session.unique():

        # Get current session data, one row per trial
        df_s = df_p[df_p.session == session]
        df_s = df_s.drop_duplicates(subset='trial', keep='first')

        # Identify the last row of the 'baseline' condition
        last_baseline_row = df_s[df_s['condition'] == 'baseline'].iloc[-1]

        # Filter out all 'baseline' rows
        df_s = df_s[df_s['condition'] != 'baseline']

        # Prepend the last baseline row to the start of df_s
        df_s = pd.concat([last_baseline_row.to_frame().T, df_s],
                         ignore_index=True)

        n_trials = len(df_s)
        confidence = df_s.confidence.values
        feedback = df_s.feedback.values

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
        bounds = [(alpha_bound[0], alpha_bound[1]),
                  (sigma_bound[0], sigma_bound[1])]


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
        nll = results_rw_symm[2]
        aic = results_rw_symm[3]
        bic = results_rw_symm[4]
        pseudo_r2 = results_rw_symm[5]


        trial_results = rw_symmetric_LR_trial((best_alpha, best_std),
                                              confidence,
                                              feedback,
                                              n_trials)

        nlls_trial = trial_results[0]
        model_pred = trial_results[1]
        sigma_vec = trial_results[2]
        confidence_clean = trial_results[3]

        # PLOT ----------------------------------------------------------------
        # Set plot variabels
        n_trials = 20 # task trials
        n_options = 100

        models = []
        trials = []
        probabilities = []
        options = []

        # Populate the data
        p_sum_list_rw = []
        p_sum_list_WSLS = []
        for trial in range(n_trials):

            for option in range(1, n_options + 1):
                # Random model
                models.append('Random')
                trials.append(trial)
                probabilities.append(1/n_options)
                options.append(option)

                # Bias model
                models.append('Bias')
                trials.append(trial)
                probabilities.append(prob_dists_bias[trial][option - 1])
                options.append(option)

                # WSLS model
                models.append('WSLS')
                trials.append(trial)
                probabilities.append(prob_dists_WSLS[trial][option - 1])
                options.append(option)

                # RW model
                dist = create_truncated_normal(model_pred[trial],
                                               sigma_vec[trial],
                                               1,
                                               n_options+1)

                models.append('RW')
                trials.append(trial)
                probabilities.append(dist.pdf(option))
                options.append(option)

        # Create DataFrame
        df_plot = pd.DataFrame({'model': models, 'trial': trials,
                           'probability': probabilities, 'option': options,
                           'trial_count': [i+1 for i in trials]})


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
                        ha="right", va="center")
                model_name_ = f'{model_name}'
                if row_val == 0:
                    ax.text(0.5, 1.4, model_name_,
                            transform=ax.transAxes,
                            ha="center", va="center")

            # Set fontsize for x-tick labels
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_size(16)
                ax.set_xlabel(ax.get_xlabel(), fontsize=16)


            # Save and Show the plot
            plt.savefig(f'{model_name}_plot.png',
                        bbox_inches='tight',
                        dpi=300)
            plt.show()

        # create full figure
        fig = plt.figure(figsize=(12, 8))

        # Define the grid layout
        gs = fig.add_gridspec(5, 4)  # 5 rows, 4 columns

        # Create subplots
        ax_images = [fig.add_subplot(gs[0:4, i]) for i in range(4)]

        # Loop through the axes and image paths and set img to axes
        image_paths = ['Random_plot.png', 'Bias_plot.png',
                       'WSLS_plot.png', 'RW_plot.png']
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

                blue_triangle = mlines.Line2D([], [], color='blue',
                                                marker='^', linestyle='None',
                                          markersize=10,
                                          label='Purple triangles')

                ax.legend(handles=[red_line, blue_triangle],
                          labels=['Participant Choice',
                                  'Model Prediction\nProbability'],
                          bbox_to_anchor=(1.8, 1),
                          facecolor='white',
                          framealpha=1)

        # NLL plot
        ax = fig.add_subplot(gs[4, :])     # Span fifth row, all columns
        x = range(1, n_trials+1)

        # Random model
        nlls_trail_random = np.full(20, -np.log(1/100))
        ax.plot(x, nlls_trail_random, c='blue', label='Random model')

        # Bias model
        ax.plot(x, nlls_bias, c='green', label='Biased model')

        # WSLS
        ax.plot(x, nlls_WSLS, c='r', label='WSLS')

        # RW
        ax.plot(x, nlls_trial, c='purple', label='RW model')

        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xticks(x, x)
        ax.set_xlabel('Trial')
        ax.set_ylabel('NLL')
        condition = df_s.condition.unique()[1]
        plt.suptitle(f'{condition}')

        ax.legend(bbox_to_anchor=(1, 1), facecolor='white', framealpha=1)

        plt.tight_layout()

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

