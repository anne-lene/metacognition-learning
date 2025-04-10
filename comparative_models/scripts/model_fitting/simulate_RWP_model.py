# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 19:22:25 2025

@author: carll
"""

# Simulates RWP model

# This script simulates the RWP model and investiagtes the difference between
# using a linear and non-linear influence of performance on confidence.
# Treating the influence as non-linear improves the regession fit slightly but
# does this improvment does not survive correction for complexity.

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

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

def RWP_sim (x, *args, non_linear=True):

    """
    Sim RWP model returning simulated confidence estimates
    alpha, sigma, w_RW, w_PD = x
    confidence, feedback, n_trials, performance = args
    """

    alpha, sigma, w_RW, w_PD, intercept = x
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
                # The "100" establish a threshold for when the error size
                # should be considered so large that it can be ignored.
                c_P = 100 + performance[t]

                # Using predictions of performance
                #PP = np.mean(performance[t-3:t]) # Performance prediction
                #PPE = performance[t] - PP # Performance prediction error
                #c_P = alpha*PPE # Performance influence on confidence

            print(c_P)
            # Using the mean rather than the sum of influences
            #k = 0
            #if w_PD > 0:
            #    k += 1
            #if w_RW > 0:
            #    k += 1
            #model_pred[t] = max(0, min(100, ((w_RW*c_rw[t]) + (w_PD*c_P))/k ))

            # Sum up the weighted influences
            model_pred[t] = max(0, min(100, intercept + (w_RW*c_rw[t]) + (w_PD*c_P)))

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


def RW_sim(x, *args):

    """
    Sim RW model and return simulated confidence.
    alpha, sigma = x
    confidence, feedback, n_trials = args
    """

    alpha, sigma = x
    confidence, feedback, n_trials = args

    model_mean = np.zeros(n_trials)
    conf_vec = np.zeros(n_trials)
    conf_dist_vec = np.zeros((n_trials,101))
    for t in range(n_trials):

        if t == 0:

            # Consider this a baseline trial
            model_mean[t] = confidence[t]

            # Calculate probabilities and simulate confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_mean[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]
            conf_dist_vec[t] = y_values

        if t > 0:

            # Get previous confidence estimate and feedback
            f = feedback[t-1]
            c = model_mean[t-1]

            # Update rule
            c_pred = c + alpha*(f - c)

            # Encure c_pred is between 0 and 100.
            c_pred = max(0, min(100, c_pred))
            model_mean[t] = c_pred

            # Calculate probabilities and simulate confidence estimate
            conf_sim, y_values = sim_norm_prob_vectorized(
                                                    np.array([model_mean[t]]),
                                                    np.array([sigma]))

            # Save simulated estimate to confidence array
            conf_vec[t] = conf_sim[0]
            conf_dist_vec[t] = y_values

    return conf_vec, conf_dist_vec

#%% Simulate linear model confidence over trials
trials = 15
iterations = 100

confidence, feedback, n_trials, performance = rand_uni_arr(), rand_uni_arr(), 15, rand_uni_arr(high=20)
performance = -performance

# Parameters
alpha, sigma, w_RW, w_P, intercept = 0.8, 3, 0.9, 0.3, -35


# Predictions
conf_vec, p_options = RWP_sim((alpha, sigma, w_RW, w_P, intercept),
                               confidence, feedback, n_trials, performance,
                               non_linear=False)

conf_vec_rw , p_options_rw  = RWP_sim((alpha, sigma, w_RW, 0, 0),
                               confidence, feedback, n_trials, performance,
                               non_linear=False)

conf_vec_p, p_options_p = RWP_sim((alpha, sigma, 0, w_P, intercept),
                               confidence, feedback, n_trials, performance,
                               non_linear=False)

# Plot
fig, (ax, ax2) = plt.subplots(1,2, figsize=(6,4))
for trial in range(5,6):
    ax.plot(p_options[trial], label='rwp', color='red', ls='-')
    ax.plot(p_options_rw[trial], label='rw', color='C1', ls='--')
    ax.plot(p_options_p[trial], label='p', color='C2', ls='--')

ax2.plot(range(len(conf_vec)), conf_vec, label='rwp', color='red')
ax2.plot(range(len(conf_vec_rw)), conf_vec_rw, label='rw', color='C1')
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
ax2.set_title('confidence over trials')
ax.set_ylim(0, 0.4)
ax2.set_ylim(0, 100)
plt.show()


#%% Simulate linear model
trials = 15
iterations = 100
p_array = np.zeros((iterations, trials))
c_array = np.zeros((iterations, trials))
for i in range(iterations):
    # Values
    #np.random.seed(12)
    confidence, feedback, n_trials, performance = rand_uni_arr(), rand_uni_arr(), 15, rand_uni_arr(high=100)
    performance = -performance

    # Parameters
    alpha, sigma, w_RW, w_P = 0.5, 25, 0.3, .9

    # Predictions
    conf_vec, p_options = RWP_sim((alpha, sigma, w_RW, w_P),
                                   confidence, feedback, n_trials, performance,
                                   non_linear=False)

    c_array[i] = conf_vec
    p_array[i] = performance
    print(i)

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,4))
#ax.plot(range(n_trials), confidence, label='data')
#ax.plot(c_array, p_array, label='sim')
ax.scatter(c_array, p_array, label='sim')
ax.set_xlabel('Confidence estimated')
ax.set_ylabel('Performance')
ax.spines[['top', 'right']].set_visible(False)
#plt.legend()
ax.set_ylim(-100,0)
plt.show()

#%% Simulate non-linear influence
trials = 15
iterations = 100
p_array = np.zeros((iterations, trials))
c_array = np.zeros((iterations, trials))
for i in range(iterations):
    # Values
    #np.random.seed(12)
    confidence, feedback, n_trials, performance = rand_uni_arr(), rand_uni_arr(), 15, rand_uni_arr(high=100)
    performance = -performance

    # Parameters
    alpha, sigma, w_RW, w_P = 0.5, 25, 0.3, .9

    # Predictions
    conf_vec, p_options = RWP_sim((alpha, sigma, w_RW, w_P),
                                   confidence, feedback, n_trials, performance,
                                   non_linear=True)

    c_array[i] = conf_vec
    p_array[i] = performance
    print(i)

# Plot
fig, ax = plt.subplots(1,1, figsize=(6,4))
#ax.plot(range(n_trials), confidence, label='data')
#ax.plot(c_array, p_array, label='sim')
ax.scatter(c_array, p_array, label='sim')
ax.set_xlabel('Confidence estimated')
ax.set_ylabel('Performance')
ax.spines[['top', 'right']].set_visible(False)
#plt.legend()
ax.set_ylim(-100,0)
plt.show()

#%% Compare with data

os.environ['R_HOME'] = r'C:\Users\carll\anaconda3\envs\metacognition_and_mood\lib\R'
from src.utility_functions import add_session_column

# Import data - Varied feedback condition (Experiment 2)
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
project_path = grandparent_directory
experiment_data_path = r'variable_feedback/data'
data_file = r'variable_fb_data_full_processed.csv'
full_path = os.path.join(project_path, experiment_data_path, data_file)
df = pd.read_csv(full_path, low_memory=False)

# =============================================================================
# # Import data - Fixed feedback condition (Experiment 1)
# data_file = r'main-20-12-14-processed_filtered.csv'
# full_path = data_file
# df = pd.read_csv(full_path, low_memory=False)
# =============================================================================

df = df.groupby('pid').apply(add_session_column).reset_index(drop=True)

# Get session data in a list
session_dfs = get_session_dfs(df)

# Loop over list and get confidence and performance
trials = 15
c_array = np.zeros((len(session_dfs), trials))
p_array = np.zeros((len(session_dfs), trials))
for i, session_df in enumerate(session_dfs):
    c_array[i], p_array[i] = process_session(session_df)


# Plot
fig, ax = plt.subplots(1,1, figsize=(6,4))
ax.scatter(c_array, p_array, label='sim')
ax.set_xlabel('Confidence')
ax.set_ylabel('Performance')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(-100, 0)
plt.show()

#%% Linear or non-linear?

X = c_array.flatten()  # confidence
y = p_array.flatten()  # performance

# Reshape X for sklearn
X_lin = X.reshape(-1, 1)
lin_model = LinearRegression().fit(X_lin, y)
y_lin_pred = lin_model.predict(X_lin)

mse_lin = mean_squared_error(y, y_lin_pred)
r2_lin = r2_score(y, y_lin_pred)

poly = PolynomialFeatures(degree=2)  # Try degree=2 or 3
X_poly = poly.fit_transform(X_lin)
poly_model = LinearRegression().fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

mse_poly = mean_squared_error(y, y_poly_pred)
r2_poly = r2_score(y, y_poly_pred)

print(f"Linear fit:     MSE = {mse_lin:.3f}, R² = {r2_lin:.3f}")
print(f"Polynomial fit: MSE = {mse_poly:.3f}, R² = {r2_poly:.3f}")

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(X, y, alpha=0.3, label='data')

# Sort X for smooth curves
sorted_idx = np.argsort(X)
X_sorted = X_lin[sorted_idx]

ax.plot(X_sorted, y_lin_pred[sorted_idx], color='red', label='Linear fit')
ax.plot(X_sorted, y_poly_pred[sorted_idx], color='green', label='Poly fit (deg=2)')

ax.set_xlabel("Confidence")
ax.set_ylabel("Performance")
ax.set_ylim(-100, 0)
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()

aic_lin = aic(y, y_lin_pred, k=2)      # Linear: 2 params (intercept + slope)
aic_poly = aic(y, y_poly_pred, k=3)    # Poly deg=2: 3 params (x², x, intercept)

print(f"AIC (Linear): {aic_lin:.2f}")
print(f"AIC (Poly):   {aic_poly:.2f}")

#%% Across Session Comparison

# Store counts and diffs
n_sessions = len(session_dfs)
wins_linear = 0
wins_poly = 0
aic_diffs = []  # poly - linear
aic_pairs = []
r2_diffs = []
mse_diffs = []

for i, session_df in enumerate(session_dfs):
    c, p = process_session(session_df)
    X = np.array(c).reshape(-1, 1)
    y = np.array(p)

    # Linear fit
    lin_model = LinearRegression().fit(X, y)
    y_lin_pred = lin_model.predict(X)
    aic_lin = aic(y, y_lin_pred, k=2)

    # Polynomial (deg=2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    poly_model = LinearRegression().fit(X_poly, y)
    y_poly_pred = poly_model.predict(X_poly)
    aic_poly = aic(y, y_poly_pred, k=3)

    aic_pairs.append((aic_lin, aic_poly))
    aic_diffs.append(aic_poly - aic_lin)  # positive means linear better

    if aic_poly < aic_lin:
        wins_poly += 1
    else:
        wins_linear += 1

    r2_lin = r2_score(y, y_lin_pred)
    r2_poly = r2_score(y, y_poly_pred)
    r2_diffs.append(r2_poly - r2_lin)

    mse_lin = mean_squared_error(y, y_lin_pred)
    mse_poly = mean_squared_error(y, y_poly_pred)
    mse_diffs.append(mse_poly - mse_lin)  # positive = poly is better

# Summary
print("\n===== Summary =====")
print(f"Mean ΔR²: {np.mean(r2_diffs):.5f}")
print(f"Mean ΔMSE: {np.mean(mse_diffs):.5f}")
print(f"Sessions better fit by linear:     {wins_linear}/{n_sessions}")
print(f"Sessions better fit by polynomial: {wins_poly}/{n_sessions}")

avg_diff = np.mean(aic_diffs)
print(f"Mean AIC difference (poly - linear): {avg_diff:.2f}")

plt.hist(r2_diffs, bins=20, edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("R² difference (poly - linear)")
plt.ylabel("Number of sessions")
plt.title("R² Difference per Session (Positive = Poly Better)")
plt.grid(True)
plt.show()

plt.hist(mse_diffs, bins=20, edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("MSE difference (poly - linear)")
plt.ylabel("Number of sessions")
plt.title("MSE Difference per Session (Negative = Poly Better)")
plt.grid(True)
plt.show()

plt.hist(aic_diffs, bins=20, edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("AIC difference (poly - linear)")
plt.ylabel("Number of sessions")
plt.title("AIC Difference per Session (Negative = Poly Better)")
plt.grid(True)
plt.show()



