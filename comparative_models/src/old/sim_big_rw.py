# -*- coding: utf-8 -*-
"""
Created on Thu May  2 02:02:44 2024

@author: carll
"""

import numpy as np
from matplotlib import pyplot as plt

def normalize_std_integrated(std_integrated, std_p_belief, s):
    min_std_integrated = std_p_belief - 100 * w
    max_std_integrated = std_p_belief + 100 * w
    normalized_std_integrated = ((std_integrated - min_std_integrated) / (max_std_integrated - min_std_integrated)) * 100
    return normalized_std_integrated

def rescale_std_integrated(std_integrated, s):
    normalized_std_integrated = std_integrated * s
    return normalized_std_integrated

def big_rw_sim (x, *args):

    """
    Simulates choice and confidence reports in dot counting task.
    """

    alpha_plus, alpha_minus, mean_p_belief, std_p_belief, w, util_func_bias, s = x
    (n_trials, n_dots,
     condition, min_value,
     max_value) = args

    confidence_pred = np.zeros(n_trials)
    fb_hist_belief = np.zeros(n_trials)
    choice_pred = np.zeros(n_trials*3)
    std_integrated = np.zeros(n_trials*3)
    subtrials = range(3)
    ct = -1
    for t in range(n_trials):

        # loop over the current trial's subtrials
        trial_error = np.zeros(3)
        for st in subtrials:
            # Choice trial number
            ct += 1

            # The number of dots believed to be seen
            perceptual_beleif = n_dots[ct] + mean_p_belief
            #perceptual_beleif = np.random.normal(n_dots[ct] + mean_p_belief, 4)
            # Integrate belief with feedback history belief
            if t==0:
                std_integrated[ct] = std_p_belief
            else:
                #std_p_belief_ = np.random.normal(std_p_belief, 4)
                #std_p_belief = std_p_belief + n_dots[ct]/100
                std_integrated[ct]  = ((1-w)*std_p_belief) + w*(s*(100-fb_hist_belief[t-1]))
                #std_integrated[ct] = std_p_belief + w*(50-fb_hist_belief[t-1])
                std_integrated[ct] =  max(0, min(std_integrated[ct], 100))

            # Make choice based on integrated belief
            #choice_pred[ct] = int(np.random.uniform(perceptual_beleif-std_integrated[ct], perceptual_beleif+std_integrated[ct]))
            choice_pred[ct] = int(np.random.normal(perceptual_beleif, std_integrated[ct]))

            # Compute error
            trial_error[st] = abs(n_dots[ct]-choice_pred[ct])



        # Report confidence: a scalar of std_integrated
# =============================================================================
#         confidence_pred[t] = max(0, min(100 - normalize_std_integrated(
#                                                             std_integrated[ct],
#                                                             std_p_belief,
#                                                             w),100))
# =============================================================================

        confidence_pred[t] = rescale_std_integrated(std_integrated[ct], s)
        #100-(((e - min_value) / (max_value - min_value)) * 100)
        #max(0, min(100, 100 - std_integrated*s))

        # Compute feedback
        e = np.mean(trial_error)
        feedback = np.random.normal(100-(((e - min_value) /
                                          (max_value - min_value)) * 100),
                                    100) # Noise

        feedback = feedback + condition
        if util_func_bias:
            feedback = feedback + float(((feedback/100)*util_func_bias))

        if feedback > confidence_pred[t]:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_plus*(feedback -
                                                                confidence_pred[t])
        else:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_minus*(feedback -
                                                                confidence_pred[t])

    return [choice_pred, confidence_pred, fb_hist_belief, std_integrated]

# Parameters and Simulation setup
n_trials = 20
n_dots = np.random.randint(50, 150, size=(n_trials*3))
condition = 0
min_value = 0
max_value = 100
x = [0.5,      # alpha_plus
     0.5,      # alpha_minus
     -10,      # mean_p_belief
     10,       # std_p_belief
     0.1,      # weight perception vs feedback
     0,        # util_func_bias
     1.,       # scaling factor
     ]

# Running the simulation
choices, confidences, feedback_histories, std_integrated = big_rw_sim(x,
                                                                      n_trials,
                                                                      n_dots,
                                                                  condition,
                                                                  min_value,
                                                                  max_value)

# Plotting the results
fig, ax = plt.subplots(5, 1, figsize=(6, 10))
ax[0].plot(choices, 'o-')
ax[0].set_title('Choice Predictions')
ax[0].set_xlabel('Trial')
ax[0].set_ylabel('Choice')

ax[1].plot(confidences, 's-')
ax[1].set_title('Confidence Predictions')
ax[1].set_xlabel('Trial')
ax[1].set_ylabel('Confidence')
ax[1].set_ylim(0,100)

ax[2].plot(feedback_histories, '^-')
ax[2].set_title('Feedback History Beliefs')
ax[2].set_xlabel('Trial')
ax[2].set_ylabel('Feedback History')
ax[2].set_ylim(0,100)


error = abs(n_dots-choices)
ax[3].plot(error, '^-')
ax[3].set_title('Choice Error')
ax[3].set_xlabel('Trial')
ax[3].set_ylabel('Error')
ax[3].set_ylim(0,150)

ax[4].plot(std_integrated, '^-')
ax[4].set_title('std_integrated')
ax[4].set_xlabel('Trial')
ax[4].set_ylabel('std_integrated')
ax[4].set_ylim(0,150)

for ax in [ax[0], ax[1], ax[2], ax[3], ax[4]]:
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
def normalize_std_integrated(std_integrated, std_p_belief, w):
    min_std_integrated = std_p_belief - 50 * w
    max_std_integrated = std_p_belief + 50 * w
    normalized_std_integrated = ((std_integrated - min_std_integrated) / (max_std_integrated - min_std_integrated)) * 100
    return normalized_std_integrated

def big_rw (x, *args):

    """
    Simulates choice and confidence reports in dot counting task.
    """

    alpha_plus, alpha_minus, mean_p_belief, std_p_belief, w, util_func_bias = x
    (n_trials,
     n_dots,
     min_value,
     max_value,
     confidence,
     feedback) = args

    confidence_pred = np.zeros(n_trials)
    fb_hist_belief = np.zeros(n_trials)
    choice_pred = np.zeros(n_trials*3)
    std_integrated = np.zeros(n_trials*3)
    sigma_vec = np.zeros(n_trials)
    subtrials = range(3)
    ct = -1
    for t in range(n_trials):

        # loop over the current trial's subtrials
        trial_error = np.zeros(3)
        for st in subtrials:
            # Choice trial number
            ct += 1

            # The number of dots believed to be seen
            perceptual_beleif = n_dots[ct] + mean_p_belief

            # Integrate belief with feedback history belief
            if t==0:
                std_integrated[ct] = std_p_belief
            else:
                #std_integrated = ((1-w)*std_p_belief) + (w*(fb_hist_belief[t-1]))
                #std_p_belief = np.random.normal(std_p_belief, 2)
                std_integrated[ct] = std_p_belief + w*(50-fb_hist_belief[t-1])
                std_integrated[ct] =  max(0, min(std_integrated[ct], 1000))

            # Make choice based on integrated belief
            #choice_pred[ct] = int(np.random.uniform(perceptual_beleif-std_integrated[ct], perceptual_beleif+std_integrated[ct]))
            choice_pred[ct] = int(np.random.normal(perceptual_beleif, std_integrated[ct]))

            # Compute error
            trial_error[st] = abs(n_dots[ct]-choice_pred[ct])

        # Report confidence: a scalar of std_integrated
        confidence_pred[t] = max(0, min(100 - normalize_std_integrated(std_integrated[ct],
                                                            std_p_belief, w),100))

        # Compute feedback
        if util_func_bias:
            feedback = feedback + float(((feedback/100)*util_func_bias))

        if feedback > confidence_pred[t]:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_plus*(feedback -
                                                                confidence_pred[t])
        else:
            # Update feedback history belief
            fb_hist_belief[t] = confidence_pred[t] + alpha_minus*(feedback -
                                                                confidence_pred[t])

        # Small value to avoid division by 0
        epsilon = 1e-10

        # Remove initial trial
        confidence_pred = confidence_pred[1:]
        sigma_vec = sigma_vec[1:]
        confidence = confidence[1:]

        # Get NLLs from truncated probability distribution
        nlls = -np.log(create_truncated_normal(model_pred,  # ignore first trial
                                               sigma_vec,
                                               lower_bound=0,
                                               upper_bound=100).pdf(confidence)
                       + epsilon)

    return [nlls, confidence_pred, sigma_vec, confidence]

    #return [choice_pred, confidence_pred, fb_hist_belief, std_integrated]





