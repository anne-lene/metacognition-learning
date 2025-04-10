# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:58:03 2024

@author: carll
"""

# Simulates how RW and CK values are combined in the RWCK model. A takeaway is
# that the beta bounds needs to be sufficiently high during model fitting to
# allow the model to be as flexible as it can be.


import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

def calc_norm_prob_vectorized(confidence, mean, sigma,
                              lower_bound=0, upper_bound=100):
    """
    Calculate the probability of confidence values given a normal distribution
    with corresponding mean and sigma values for each point. Ensure that the
    probability density values for integer steps between lower_bound and upper_bound
    sum to 1 for each distribution, using vectorized operations.

    Parameters:
    confidence (array): List or array of confidence values.
    mean (array): List or array of mean values corresponding to each confidence value.
    sigma (array): List or array of standard deviation values corresponding to each confidence value.
    lower_bound (int): The lower bound for the range of values to normalize over (default 0).
    upper_bound (int): The upper bound for the range of values to normalize over (default 100).

    Returns:
    numpy.ndarray: An array of probabilities for each confidence[i] based on
                   the normalized normal distribution defined by mean[i] and sigma[i].
    """

    # Create an array of integer steps [lower_bound, upper_bound] for all distributions
    x_values = np.arange(lower_bound, upper_bound + 1)

    # Reshape mean and sigma to broadcast across x_values (for element-wise operations)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]

    # Compute the Gaussian probability for each x_value, mean, and sigma (vectorized)
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)

    # Normalize each distribution by dividing each by the sum of the distribution
    y_values /= np.sum(y_values, axis=1, keepdims=True)

    # Extract probabilities for the confidence values (vectorized)
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]

    return probabilities, y_values

# Simulate some example data for visualization
n_trials = 15
array_length = 101
alpha = 0.0
alpha_ck = 0.3
sigma = 10
sigma_ck = sigma
beta = 0
beta_ck = 2
bias = 100

feedback = np.random.choice([0, 1], size=n_trials)
feedback = np.random.randint(0, 101, size=n_trials)
confidence = np.random.randint(0, 101, size=n_trials)

# Initialize choice kernels
choice_kernels = np.full(array_length, 1 / array_length)
model_pred = np.zeros(n_trials)
probabilities_list = []

# Example trial to visualize
trial_to_visualize = 8

for t in range(n_trials):

    if t == 0:
        model_pred[t] = bias

    else:
        # Rescorla-Wagner Update Rule (RW)
        f = feedback[t-1]
        c = model_pred[t-1]
        PE = f - c  # Prediction error
        c_pred = c + alpha * PE
        c_pred = max(0, min(100, c_pred))

        _, p_options = calc_norm_prob_vectorized(np.array([confidence[t]]),
                                                 np.array([c_pred]),
                                                 np.array([sigma]),

                                                 lower_bound=0,
                                                 upper_bound=100)

        p_options = p_options[0]

        # Update choice kernels based on previous choices
        for k in range(101):
            a_k_t = 1 if k == c else 0
            #choice_kernels[k] += alpha_ck * (a_k_t - choice_kernels[k])
            choice_kernels[k] = choice_kernels[k] + alpha_ck * (a_k_t - choice_kernels[k])

        # Apply Gaussian smoothing to choice kernels
        smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma_ck)

        # Normalise the choice kernel
        smoothed_choice_kernels = smoothed_choice_kernels / np.sum(smoothed_choice_kernels)

        # Combine RW update (p_options) and choice kernels (smoothed_choice_kernels)
        combined_p = np.zeros(len(p_options))
        for i, (v_k, ck_k) in enumerate(zip(p_options, smoothed_choice_kernels)):
            combined_p[i] = np.exp((beta * v_k) + (beta_ck * ck_k))

        # Normalize combined probabilities
        combined_probabilities = combined_p / np.sum(combined_p)
        #combined_probabilities =  smoothed_choice_kernels
        probabilities_list.append(combined_probabilities)

        model_pred[t] = np.argmax(combined_probabilities)

        # Visualize at a specific trial (e.g., trial 5)
        if t < trial_to_visualize:

            fig, ax = plt.subplots(1,1, figsize=(12, 6))

            # Plot RW update probabilities (p_options)
            plt.plot(np.arange(101), p_options, label="RW Update (p_options)",
                     linestyle="--", color="blue")

            # Plot smoothed choice kernels
            plt.plot(np.arange(101), smoothed_choice_kernels,
                     label="Smoothed Choice Kernels", linestyle=":",
                     color="green")

            # Plot combined probabilities
            plt.plot(np.arange(101), combined_probabilities,
                     label="Combined Probabilities", linestyle="-",
                     color="red")

            plt.xlabel("Option (0 to 100)")
            plt.ylabel("Probability")
            plt.title(f"Probability Distribution at Trial {t}")
            plt.legend()
            ax.spines[['top', 'right']].set_visible(False)
            plt.show()


            print('rw sum', sum(p_options))
            print('kernel sum', sum(smoothed_choice_kernels))
            print('combined_probabilities sum', sum(combined_probabilities))

#%% Parameter recovery

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d


# Same calc_norm_prob_vectorized function as before
def calc_norm_prob_vectorized(confidence, mean, sigma,
                              lower_bound=0, upper_bound=100):
    x_values = np.arange(lower_bound, upper_bound + 1)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)
    y_values /= np.sum(y_values, axis=1, keepdims=True)
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]
    return probabilities, y_values

# Model simulation function
def simulate_model(params, n_trials=30):
    alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias = params
    array_length = 101

    feedback = np.random.randint(0, 101, size=n_trials)
    confidence = np.random.randint(0, 101, size=n_trials)

    choice_kernels = np.full(array_length, 1 / array_length)
    model_pred = np.zeros(n_trials)
    choices = np.zeros(n_trials)

    for t in range(n_trials):
        if t == 0:
            model_pred[t] = bias
        else:
            f = feedback[t-1]
            c = model_pred[t-1]
            PE = f - c
            c_pred = c + alpha * PE
            c_pred = max(0, min(100, c_pred))

            _, p_options = calc_norm_prob_vectorized(np.array([confidence[t]]),
                                                     np.array([c_pred]),
                                                     np.array([sigma]))
            p_options = p_options[0]

            # Update choice kernels based on previous choices
            for k in range(101):
                a_k_t = 1 if k == c else 0
                choice_kernels[k] = choice_kernels[k] + alpha_ck * (a_k_t - choice_kernels[k])

            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma_ck)
            smoothed_choice_kernels = smoothed_choice_kernels / np.sum(smoothed_choice_kernels)

            combined_p = np.zeros(len(p_options))
            for i, (v_k, ck_k) in enumerate(zip(p_options, smoothed_choice_kernels)):
                combined_p[i] = np.exp((beta * v_k) + (beta_ck * ck_k))
            combined_probabilities = combined_p / np.sum(combined_p)

            choices[t] = np.argmax(combined_probabilities)
            model_pred[t] = choices[t]

    return choices, feedback, confidence

# Likelihood function
def likelihood(params, choices, feedback, confidence):
    alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias = params
    array_length = 101
    n_trials = len(choices)
    choice_kernels = np.full(array_length, 1 / array_length)
    model_pred = np.zeros(n_trials)

    log_likelihood = 0

    for t in range(n_trials):
        if t == 0:
            model_pred[t] = bias
        else:
            f = feedback[t-1]
            c = model_pred[t-1]
            PE = f - c
            c_pred = c + alpha * PE
            c_pred = max(0, min(100, c_pred))

            _, p_options = calc_norm_prob_vectorized(np.array([confidence[t]]),
                                                     np.array([c_pred]),
                                                     np.array([sigma]))
            p_options = p_options[0]

            for k in range(101):
                a_k_t = 1 if k == c else 0
                choice_kernels[k] = choice_kernels[k] + alpha_ck * (a_k_t - choice_kernels[k])

            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma_ck)
            smoothed_choice_kernels = smoothed_choice_kernels / np.sum(smoothed_choice_kernels)

            combined_p = np.zeros(len(p_options))
            for i, (v_k, ck_k) in enumerate(zip(p_options, smoothed_choice_kernels)):
                combined_p[i] = np.exp((beta * v_k) + (beta_ck * ck_k))

            combined_probabilities = combined_p / np.sum(combined_p)
            chosen_option = int(choices[t])

            log_likelihood += np.log(combined_probabilities[chosen_option] + 1e-8)  # Add small value to avoid log(0)

    return -log_likelihood  # Negative because we minimize in optimization


# Parameter recovery script
def perform_parameter_recovery():
    true_params = [0.9, 0.1, 10, 10, 10, 80, 80]  # [alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias]
    choices, feedback, confidence = simulate_model(true_params)

    # Initial guesses for fitting
    initial_guess = [0.5, 0.2, 5, 5, 5, 50, 50]

    # Bounds for the parameters to be recovered
    bounds = [(0, 1), (0, 1), (1, 20), (1, 20), (1, 50), (1, 100), (0, 100)]

    # Use minimize function from scipy to recover parameters
    result = minimize(likelihood, initial_guess, args=(choices, feedback, confidence), bounds=bounds, method='L-BFGS-B')

    print("True parameters:", true_params)
    print("Recovered parameters:", result.x)

    return result

# Run parameter recovery
perform_parameter_recovery()

#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_absolute_error

# Function definitions (same as before)
def calc_norm_prob_vectorized(confidence, mean, sigma,
                              lower_bound=0, upper_bound=100):
    x_values = np.arange(lower_bound, upper_bound + 1)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)
    y_values /= np.sum(y_values, axis=1, keepdims=True)
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]
    return probabilities, y_values

def simulate_model(params, n_trials=30):
    alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias = params
    array_length = 101
    sigma = sigma_ck
    feedback = np.random.randint(0, 101, size=n_trials)
    confidence = np.random.randint(0, 101, size=n_trials)

    choice_kernels = np.full(array_length, 1 / array_length)
    model_pred = np.zeros(n_trials)
    choices = np.zeros(n_trials)

    for t in range(n_trials):
        if t == 0:
            model_pred[t] = bias
        else:
            f = feedback[t-1]
            c = model_pred[t-1]
            PE = f - c
            c_pred = c + alpha * PE
            c_pred = max(0, min(100, c_pred))

            _, p_options = calc_norm_prob_vectorized(np.array([confidence[t]]),
                                                     np.array([c_pred]),
                                                     np.array([sigma]))
            p_options = p_options[0]

            for k in range(101):
                a_k_t = 1 if k == c else 0
                choice_kernels[k] = choice_kernels[k] + alpha_ck * (a_k_t - choice_kernels[k])

            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma_ck)
            smoothed_choice_kernels = smoothed_choice_kernels / np.sum(smoothed_choice_kernels)

            combined_p = np.zeros(len(p_options))
            for i, (v_k, ck_k) in enumerate(zip(p_options, smoothed_choice_kernels)):
                combined_p[i] = np.exp((beta * v_k) + 0*(beta_ck * ck_k))

            # Normalize combined probabilities correctly
            combined_probabilities = combined_p / np.sum(combined_p)

            choices[t] = np.random.choice(np.arange(
                                                len(combined_probabilities)
                                                ),
                                          p=combined_probabilities)
            model_pred[t] = choices[t]

    return choices, feedback, confidence

def likelihood(params, choices, feedback, confidence):
    alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias = params
    sigma = sigma_ck
    array_length = 101
    n_trials = len(choices)
    choice_kernels = np.full(array_length, 1 / array_length)
    model_pred = np.zeros(n_trials)

    log_likelihood = 0

    for t in range(n_trials):
        if t == 0:
            model_pred[t] = bias
        else:
            f = feedback[t-1]
            c = model_pred[t-1]
            PE = f - c
            c_pred = c + alpha * PE
            c_pred = max(0, min(100, c_pred))

            _, p_options = calc_norm_prob_vectorized(np.array([confidence[t]]),
                                                     np.array([c_pred]),
                                                     np.array([sigma]))
            p_options = p_options[0]

            for k in range(101):
                a_k_t = 1 if k == c else 0
                choice_kernels[k] = choice_kernels[k] + alpha_ck * (a_k_t - choice_kernels[k])

            smoothed_choice_kernels = gaussian_filter1d(choice_kernels, sigma_ck)
            smoothed_choice_kernels = smoothed_choice_kernels / np.sum(smoothed_choice_kernels)

            combined_p = np.zeros(len(p_options))
            for i, (v_k, ck_k) in enumerate(zip(p_options, smoothed_choice_kernels)):
                combined_p[i] = np.exp((beta * v_k) + 0*(beta_ck * ck_k))

            combined_probabilities = combined_p / np.sum(combined_p)
            chosen_option = int(choices[t])

            log_likelihood += np.log(combined_probabilities[chosen_option] + 1e-8)

    return -log_likelihood  # Negative because we minimize in optimization


# Perform parameter recovery over a range for each parameter
def perform_parameter_recovery_over_range(param_index, param_range,
                                          true_params, n_trials=30):
    recovered_params = []

    for param_value in param_range:
        params = true_params.copy()
        params[param_index] = param_value  # Set the parameter we're testing

        # Simulate the data with this parameter value
        choices, feedback, confidence = simulate_model(params, n_trials)

        # Initial guesses for fitting
        initial_guess = [0.5, 0.2, 5, 5, 5, 50, 50]
        bounds = [(0, 1), (0, 1), # alpha
                  (1, 100), (1, 100),  # sigma
                  (1, 200), (1, 200),  # beta
                  (0, 100)] # bias

        # Recover the parameters using optimization
        result = minimize(likelihood, initial_guess,
                          args=(choices, feedback, confidence),
                          bounds=bounds, method='L-BFGS-B')

        # Store the recovered parameter
        recovered_params.append(result.x[param_index])

    return np.array(recovered_params)

# Range of parameter values for testing
param_ranges = {
    'alpha': np.linspace(0.1, 1.0, 10),
    'alpha_ck': np.linspace(0.1, 1.0, 10),
    'sigma': np.linspace(1, 100, 100),
    'sigma_ck': np.linspace(1, 100, 100),
    'beta': np.linspace(1, 200, 200),
    'beta_ck': np.linspace(1, 200, 200),
    'bias': np.linspace(0, 100, 100)
}

# Perform recovery for each parameter
def perform_recovery_for_all_params():
    # these values are overwritten when looping over range.
    true_params = [0.9, 0.1, 10, 10, 10, 80, 80]  # [alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias]
    param_names = ['alpha', 'alpha_ck', 'sigma',
                   'sigma_ck', 'beta', 'beta_ck', 'bias']

    for i, param_name in enumerate(param_names):
        param_range = param_ranges[param_name]
        recovered_params = perform_parameter_recovery_over_range(i, param_range, true_params)

        # Plot true vs recovered parameters for each range
        plt.figure(figsize=(8, 6))
        plt.plot(param_range, param_range, 'r--', label="Ideal")
        plt.scatter(param_range, recovered_params, label=f"Recovered {param_name}")
        plt.xlabel(f"True {param_name}")
        plt.ylabel(f"Recovered {param_name}")
        plt.title(f"Parameter Recovery for {param_name}")
        plt.legend()
        plt.show()

        # Quantify recovery
        correlation = np.corrcoef(param_range, recovered_params)[0, 1]
        mae = mean_absolute_error(param_range, recovered_params)
        print(f"Correlation for {param_name}: {correlation:.4f}")
        print(f"Mean Absolute Error for {param_name}: {mae:.4f}")

# Perform parameter recovery over a range for each parameter with 10 iterations per step
def perform_parameter_recovery_over_range(param_index, param_range,
                                          true_params, n_trials=30,
                                          n_iterations=10):
    mean_recovered_params = []

    for param_value in param_range:
        params = true_params.copy()
        params[param_index] = param_value  # Set the parameter we're testing
        recovered_params_for_value = []

        # Perform n_iterations for each parameter value
        for _ in range(n_iterations):
            # Simulate the data with this parameter value
            choices, feedback, confidence = simulate_model(params, n_trials)

            # Initial guesses for fitting
            initial_guess = [0.5, 0.2, 5, 5, 5, 50, 50]
            bounds = [(0, 1), (0, 1), # alpha
                      (1, 100), (1, 100),  # sigma
                      (1, 200), (1, 200),  # beta
                      (0, 100)] # bias


            # Recover the parameters using optimization
            result = minimize(likelihood, initial_guess,
                              args=(choices, feedback, confidence),
                              bounds=bounds, method='L-BFGS-B')

            # Store the recovered parameter for this iteration
            recovered_params_for_value.append(result.x[param_index])

        # Take the mean of the recovered parameters over the 10 iterations
        mean_recovered_params.append(np.mean(recovered_params_for_value))

    return np.array(mean_recovered_params)

# Perform recovery for each parameter
def perform_recovery_for_all_params(n_iterations=10):
    true_params = [0.9, 0.1, 10, 10, 10, 80, 80]  # [alpha, alpha_ck, sigma, sigma_ck, beta, beta_ck, bias]
    param_names = ['alpha', 'alpha_ck', 'sigma', 'sigma_ck', 'beta', 'beta_ck', 'bias']

    for i, param_name in enumerate(param_names):
        param_range = param_ranges[param_name]
        recovered_params = perform_parameter_recovery_over_range(i,
                                                                 param_range,
                                                                 true_params,
                                                                 n_iterations=n_iterations)

        # Plot true vs recovered parameters for each range
        plt.figure(figsize=(8, 6))
        plt.plot(param_range, param_range, 'r--', label="Ideal")
        plt.scatter(param_range, recovered_params, label=f"Recovered {param_name}")
        plt.xlabel(f"True {param_name}")
        plt.ylabel(f"Recovered {param_name}")
        plt.title(f"Parameter Recovery for {param_name}")
        plt.legend()
        plt.show()

        # Quantify recovery
        correlation = np.corrcoef(param_range, recovered_params)[0, 1]
        mae = mean_absolute_error(param_range, recovered_params)
        print(f"Correlation for {param_name}: {correlation:.4f}")
        print(f"Mean Absolute Error for {param_name}: {mae:.4f}")

# Run parameter recovery for all parameters
perform_recovery_for_all_params()
