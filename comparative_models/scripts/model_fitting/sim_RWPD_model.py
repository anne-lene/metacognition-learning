# -*- coding: utf-8 -*-
"""
Created on Sat Apr  5 15:49:53 2025

@author: carll
"""

# Simulate RWPD model

# Simualte the RWPD model at different paramters to get an understanding of
# how they influence each other.

import numpy as np
from matplotlib import pyplot as plt


def sim_norm_prob_vectorized(mean, sigma,
                             lower_bound=0, upper_bound=100):
    """

    """
    #np.random.seed(42) # Make deterministic (for debugging)

    # Create an array of integer steps [lower_bound, upper_bound] for all distributions
    x_values = np.arange(lower_bound, upper_bound + 1)

    # Reshape mean and sigma to broadcast across x_values (for element-wise operations)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]

    # Compute the Gaussian probability for each x_value, mean, and sigma (vectorized)
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)

    # Normalize each distribution by dividing each by the sum of the distribution
    y_values /= np.sum(y_values, axis=1, keepdims=True)

    # Simulate confidence values by sampling based on the probabilities in y_values
    simulated_confidence = np.array([
        np.random.choice(x_values, p=prob_dist) for prob_dist in y_values
    ])

    return simulated_confidence, y_values

def RWPD_sim (x, *args):

    """
    Sim RWPD model retuning simulated confidence estimates
    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args
    """

    alpha, sigma, gamma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args

    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    prob_vec = np.zeros((n_trials, 101))
    ewma_performance = performance[0]
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial
            model_pred[t] = confidence[t]

            # Calculate probabilities and simulate confidence estimate
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                    np.array([model_pred[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]
            prob_vec[t] = p_options

        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = int(c_rw[t-1])

            PE = f - c  # Prediction error
            c_rw[t] = c + alpha*PE  # Update rule

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance delta
            #delta_p = performance[t] - np.nanmean(performance[t-gamma:t])

            # Update EWMA
            ewma_performance = gamma * ewma_performance + (1 - gamma) * performance[t-1]
            # Compute performance delta
            delta_p = performance[t] - ewma_performance


            # Encure delta_p is between -100 and 100.
            delta_p = max(-100, min(100, delta_p))

            model_pred[t] = max(0, min(100, int((w_RW*c_rw[t]) + (w_PD*delta_p))))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     lower_bound=0,
                                                     upper_bound=100)
            conf_vec[t] = conf_sim[0]
            prob_vec[t] = p_options

    return conf_vec, prob_vec


def RWP_sim (x, *args, non_linear=True):

    """
    Sim RWP model returning simulated confidence estimates
    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args
    """

    alpha, sigma, w_RW, w_PD, theta = x
    confidence, feedback, n_trials, performance = args
    model_pred = np.zeros(n_trials)
    c_rw = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    p_options_vec = np.zeros((n_trials,101))
    for t in range(n_trials):

        if t == 0:
            # The probability mean for the first trial
            model_pred[t] = confidence[t]

            # Calculate probabilities and simulate confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_pred[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]
            p_options_vec[t] = y_values

        else:
            # Get previous feedback (f) and confidence estimate (c)
            f = feedback[t-1]
            c = model_pred[t-1] #c_rw[t-1]

            # Update rule
            c_rw[t]  = c + alpha*(f - c)

            # Ensure c_rw is between 0 and 100.
            c_rw[t] = max(0, min(100, c_rw[t]))

            # Get performance influence on confidence
            if non_linear:
                c_P = 100 / (1 + np.exp(-performance[t]  / 20))
            else:
                # Negative absolute error.
                # The theta/"100" establish a threshold for when the error
                # should be considered worse than expected.
                c_P = theta + performance[t]

                # Using predictions of performance
                #PP = np.mean(performance[t-3:t]) # Performance prediction
                #PPE = performance[t] - PP # Performance prediction error
                #c_P = alpha*PPE # Performance influence on confidence

            #print(c_P)
            # Using the mean rather than the sum of influences
            #k = 0
            #if w_PD > 0:
            #    k += 1
            #if w_RW > 0:
            #    k += 1
            #model_pred[t] = max(0, min(100, ((w_RW*c_rw[t]) + (w_PD*c_P))/k ))

            # Sum up the weighted influences
            model_pred[t] = max(0, min(100, (w_RW*c_rw[t]) + (w_PD*c_P)))

            # Calculate probabilities across different options (p_options)
            conf_sim, p_options = sim_norm_prob_vectorized(
                                                     np.array([model_pred[t]]),
                                                     np.array([sigma]),
                                                     )

            conf_vec[t] = conf_sim[0]
            p_options_vec[t] = p_options

    return conf_vec, p_options_vec

def rand_uni_arr(low=0, high=100, size=15):
    array = np.random.uniform(low=low, high=high, size=(size))
    return array

def aic(y_true, y_pred, k):
    resid = y_true - y_pred
    sse = np.sum(resid**2)
    n = len(y_true)
    return n * np.log(sse / n) + 2 * k  # k = number of parameters

# Function to process data for a single session
def process_session(df_s):

    # Remove baseline
    df_s = df_s[df_s.condition != 'baseline']

    # Calculate absolute trial error as the average across subtrials
    df_s['difference'] = abs(df_s['estimate'] - df_s['correct'])
    abs_error_avg = df_s.groupby('trial')['difference'].mean()

    # Only keep first row of every subtrial (one row = one trial)
    df_s = df_s.drop_duplicates(subset='trial', keep='first')

    # Calculate trial-by-trial metrics
    confidence = df_s.confidence.values

    performance = -abs_error_avg.values # Performance from participant.

    return confidence, performance

def get_session_dfs(df):
    # Unique pid and sessions pairs
    unique_pairs = df[['pid', 'session']].drop_duplicates()

    # Convert the DataFrame to a list of tuples
    unique_pairs_list = list(unique_pairs.itertuples(index=False, name=None))
    # unique_pairs_list = [unique_pairs_list[0]]

    # Create a list of DataFrames, one for each unique pid-session combo
    df_list = [df[(df['pid'] == pid) &
                  (df['session'] == session)].copy()
               for pid, session in unique_pairs_list]

    return df_list

#%% Simulate linear model confidence over trials
trials = 15
iterations = 100

confidence, feedback, n_trials, performance = rand_uni_arr(), rand_uni_arr(), 15, rand_uni_arr(high=30)
performance = -performance

# Parameters
# The range w_PD = 2-5 balances the range w_RW = 0-1 without the model
# regressing into a simple RW model.

alpha, sigma, gamma, w_RW, w_PD, w_P, theta = 0.2, 10, 1, 0.5, 1, 0.2, 20

# Predictions
conf_vec, p_options = RWPD_sim((alpha, sigma, gamma, w_RW, w_PD),
                               confidence, feedback, n_trials, performance,
                               )

conf_vec_rw , p_options_rw  = RWPD_sim((alpha, sigma, gamma, w_RW, 0),
                               confidence, feedback, n_trials, performance,
                               )

conf_vec_p, p_options_p = RWPD_sim((alpha, sigma, gamma, 0, w_PD),
                               confidence, feedback, n_trials, performance,
                               )

# Predictions
conf_vec_rwp, p_options_rwp = RWP_sim((alpha, sigma, w_RW, w_P, theta),
                               confidence, feedback, n_trials, performance,
                               non_linear=False)

# Plot
fig, (ax, ax2) = plt.subplots(1,2, figsize=(6,4))
for trial in range(7,8):
    ax.plot(p_options[trial], label='rwpd', color='red', ls='-')
    ax.plot(p_options_rw[trial], label='rw', color='C1', ls='--')
    ax.plot(p_options_p[trial], label='pd', color='C2', ls='--')
    ax.plot(p_options_rwp[trial], label='rwp', color='C4', ls='--')

ax2.plot(range(len(conf_vec)), conf_vec, label='rwpd', color='red')
ax2.plot(range(len(conf_vec_rw)), conf_vec_rw, label='rw', color='C1')
ax2.plot(range(len(conf_vec_rwp)), conf_vec_rwp, label='rwp', color='C4')
#ax.plot(range(n_trials), confidence, label='data')
#ax.plot(c_array, p_array, label='sim')
#ax.scatter(c_array, p_array, label='sim')
ax.set_xlabel('Confidence estimated')
ax.set_ylabel('Probability')
ax.spines[['top', 'right']].set_visible(False)
ax2.spines[['top', 'right']].set_visible(False)
ax.legend()
ax2.legend()
ax.set_title(f'trial: {trial}')
ax2.set_ylabel('Confidence estimated')
ax2.set_xlabel('Trials')
ax2.set_title(f'confidence over trials')
ax.set_ylim(0, 0.4)
plt.show()
