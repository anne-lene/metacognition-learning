# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:16:51 2025

@author: carll
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def create_nonsense_corr_data(n=15, noise_std=5, drift_freq=0.1, drift_amp=5,
                              pid=0, plot=True, num_sigmoids=3):
    """
    Simulates time series data with a spurious correlation structure.

    Parameters:
    - n: Number of time points.
    - noise_std: Standard deviation of Gaussian noise.
    - drift_freq: Frequency of the sine wave for Drift.
    - drift_amp: Amplitude of the sine wave for Drift.

    Returns:
    - A dictionary containing the four time series.
    """
    # Initialize time vector
    time = np.arange(n)

    # Generate drift as a sine wave function
  #  phase_shift = np.random.uniform(0, 2 * np.pi)  # Random phase shift
  #  Drift = drift_amp * np.sin(2 * np.pi * drift_freq * time + phase_shift) # non-linear

    # Generate sigmoid-based drift
    Drift = np.zeros(n)
    for _ in range(num_sigmoids):
        t0 = np.random.uniform(0, n)  # Random center time for the sigmoid
        k = np.random.uniform(0.1, 1.0)  # Random steepness factor
        A = np.random.uniform(0.5, 1.5) * drift_amp  # Random amplitude

        sigmoid = A / (1 + np.exp(-k * (time - t0)))  # Logistic sigmoid function
        Drift += sigmoid

    # Generate pink noise using a simple 1/f filter approximation
    pink_noise = np.cumsum(np.random.normal(0, noise_std, n))  # Integrated white noise
    Drift += pink_noise * 1  # Scale pink noise contribution

   # Drift = 1 * time # linear


    # Initialize Performance, Confidence, and Feedback
    Performance = np.zeros(n)
    Confidence = np.zeros(n)
    Feedback = np.zeros(n)

    # Initialize first values randomly
    Performance[0] = np.random.uniform(0, 100) + Drift[0]
    Feedback[0] = Performance[0] + np.random.normal(0, noise_std)
    Confidence[0] = Drift[0] + np.random.normal(0, noise_std)

    # Simulate time series data
    for t in range(1, n):
        Performance[t] = np.random.uniform(0, 100) + Drift[t] # np.random.normal(0, noise_std)
        Feedback[t] = Performance[t] + np.random.normal(0, noise_std)
        Confidence[t] = Drift[t] + np.random.normal(0, noise_std)

    # Plot the time series
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(time, Performance, label="Performance", linestyle="-")
        plt.plot(time, Confidence, label="Confidence", linestyle="--")
        plt.plot(time, Feedback, label="Feedback", linestyle="-.")
        plt.plot(time, Drift, label="Drift", linestyle=":")

        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.title("Simulated Nonsense Correlation Data")
        plt.legend()
        plt.show()

    # Return the generated data
    return {
        "pid": pid,
        "Time": time,
        "Performance": Performance,
        "Confidence": Confidence,
        "Feedback": Feedback,
        "Drift": Drift
    }

# Generate and plot the simulated data
simulated_data = create_nonsense_corr_data()

#%% Generate data

df = pd.DataFrame({
    "pid": [],
    "Time": [],
    "Performance":  [],
    "Confidence": [],
    "Feedback": [],
    "Drift": []
})

for i in range(100):
    df = pd.concat([df, pd.DataFrame(create_nonsense_corr_data(pid=i,
                                                               plot=False))],
                   ignore_index=True)

#%% Make previous feedback column

df['Feedback_t_minus_1'] = df.groupby(('pid'))['Feedback'].shift(1)
df = df.dropna().reset_index(drop=True)


#%% Regression
import statsmodels.formula.api as smf

# Fit a linear mixed-effects model predicting Confidence from Performance and Feedback
model = smf.mixedlm("Confidence ~ Feedback_t_minus_1", df, groups=df["pid"], re_formula="1 + Time").fit()

# Display the model summary
model.summary()

#%% Permutation test across participants

import statsmodels.formula.api as smf
import numpy as np
import warnings

# Remove warnings
warnings.simplefilter("ignore", category=UserWarning)  # Suppress user warnings
warnings.simplefilter("ignore", category=RuntimeWarning)  # Suppress runtime warnings
warnings.simplefilter("ignore", category=FutureWarning)  # Suppress future warnings

def shuffle_confidence(df_orig):

    df = df_orig.copy()

    # Shuffle PIDs
    unique_pids = df.pid.unique()
    shuffled_pids = np.random.permutation(unique_pids)

    # Create true to shuffle mapping
    pid_mapping = dict(zip(unique_pids, shuffled_pids))

    # Create permutation df
    df_permutation = pd.DataFrame({'true_pid': df.pid,
                                   'Confidence': df.Confidence,
                                   'Feedback': df.Feedback,
                                   'Feedback_t_minus_1': df.Feedback_t_minus_1,
                                   'Time': df.Time,
                                   'shuffled_pid': [pid_mapping[i] for i in df.pid],
                                   'Shuffled_Confidence': np.zeros(len(df))})

    # Update Shuffled_Confidence
    for i, row in df_permutation.iterrows():

        df_permutation.loc[i, 'Shuffled_Confidence'] = \
              df_permutation.loc[(df_permutation['true_pid']==row.shuffled_pid) & \
                                 (df_permutation['Time']==row.Time), 'Confidence'].values[0]

    return df_permutation

# Original model fit
original_model = smf.mixedlm("Confidence ~ Feedback_t_minus_1", df,
                             groups=df["pid"], re_formula="1 + Time").fit()
original_pvalue = original_model.pvalues['Feedback_t_minus_1'] # store p value

# Number of permutations
num_permutations = 999

# Store permutation test results
perm_results = []

# Build null distribution
for iteration in tqdm(range(num_permutations)):

    # Shuffle confidence
    perm_df = shuffle_confidence(df)
    model = smf.mixedlm("Shuffled_Confidence ~ Feedback_t_minus_1", perm_df,
                                 groups=perm_df["true_pid"], re_formula="1 + Time").fit()
    p_value = model.pvalues['Feedback_t_minus_1'] # store p value

    # Store result
    perm_results.append(p_value)

# Count how many times the true p-value is below the permuted p-values
perm_results = np.array(perm_results)
count_below_permuted_pvals = np.sum(original_pvalue < perm_results)
true_p_value = 2 / (len(perm_results)+1)

#%% Plot
fig, ax= plt.subplots(1,1, figsize=(6,4))
ax.hist(perm_results, bins=100)
ax.axvline(original_pvalue, ls='-', color='r')
plt.title(f'true p-value {true_p_value}')
plt.show()

#%%
