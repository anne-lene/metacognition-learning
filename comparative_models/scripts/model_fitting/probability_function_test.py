# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 17:56:46 2024

@author: carll
"""

# This script tests several ways of computing probabilities, showing that the
# function used in the model fitting scripts return the correct probabilities,
# and does so faster than the alternative functions. The script also show how
# the distrbutions from the function can be inverted and normalised.

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import truncnorm
import random
import time

# This function will estimate probabilities close to the bounds to be higher
# or lower than they should be. The reason for this is that the function builds
# a normalised continuous distribution under which the integral is 1, but when
# only a few descrete values are indexed from this continuous curve which
# resutls in an inprecise estimate.

def create_truncated_normal(mean, sd, lower_bound=0, upper_bound=100):
    """
    Creates a truncated normal distribution within specified bounds.

    This function generates a truncated normal distribution with a given mean
    and standard deviation, truncated within specified lower and upper bounds.
    The distribution is truncated so that values lie within the given range,
    ensuring the integral of the distribution's probability density function
    is 1 within these bounds.

    Parameters:
    mean (float): The mean of the normal distribution.
    sd (float): The standard deviation of the normal distribution.
    lower_bound (float, optional): The lower boundary for truncation of the
                                   normal distribution. Default is 0.
    upper_bound (float, optional): The upper boundary for truncation of the
                                   normal distribution. Default is 100.

    Returns:
    scipy.stats.rv_continuous: A truncated normal continuous random variable
                               object as defined in scipy.stats.

    Note:
    The function adjusts the bounds to the standard normal distribution's
    z-scores before creating the truncated distribution. This distribution
    can be used to generate random variates and to compute various statistics.
    """

    # Convert the bounds to z-scores
    lower_bound_z = (lower_bound - mean) / sd
    upper_bound_z = (upper_bound - mean) / sd

    # Create the truncated normal distribution
    return truncnorm(lower_bound_z, upper_bound_z, loc=mean, scale=sd)


# Generate data
ntrials = 100
mean = 0
sigma = 1
confidence = np.array(random.sample(range(0, 101), ntrials))# generate random numbers
model_pred = np.full(ntrials, mean) #np.array(random.sample(range(0, 100), ntrials)) # generate random numbers
sigma_vec = np.full(ntrials, sigma)

# Small value to avoid division by 0
epsilon = 1e-10

# Remove initial baseline trial
model_pred = model_pred[1:]
sigma_vec = sigma_vec[1:]
confidence = confidence[1:]

start_time = time.perf_counter()  # Start the timer

# Get NLLs from truncated probability distribution
nlls = create_truncated_normal(model_pred,  # Ignore first trial
                               sigma_vec,
                               lower_bound=0,
                               upper_bound=100).pdf(confidence)

end_time = time.perf_counter()  # End the timer
execution_time = end_time - start_time  # Calculate elapsed time

print('sum of nlls:', sum(nlls))
print(f"Execution time: {execution_time:.6f} seconds")

plt.scatter(confidence, nlls)
plt.show()

#%%

# This function fixes the earlier issue by normalising the discrete options
# within the bounds. However, the function is now much slower. It is still
# using numpy broadcasting for parallel matrix operations but now also a python
# loop slowing everything down making the complexity o(n_trials x option_range)

def create_truncated_normal_new(mean, sd, lower_bound=0, upper_bound=100):
    """
    Creates a truncated normal distribution and returns discrete probability
    values for integer points within specified bounds.

    This function generates a truncated normal distribution with a given mean
    and standard deviation, truncated within specified lower and upper bounds.
    The distribution is truncated so that values lie within the given range.
    The function evaluates the probability density function (PDF) at integer
    values between the lower and upper bounds and normalizes these values so
    that their sum is 1. This makes the output suitable for discrete
    probability-based applications.

    Parameters:
    mean (float): The mean of the normal distribution.
    sd (float): The standard deviation of the normal distribution.
    lower_bound (int, optional): The lower boundary for truncation and the
                                 range of integer values to evaluate the PDF.
                                 Default is 0.
    upper_bound (int, optional): The upper boundary for truncation and the
                                 range of integer values to evaluate the PDF.
                                 Default is 100.

    Returns:
    numpy.ndarray: An array of normalized discrete probability values for
                   integers from lower_bound to upper_bound, based on the
                   truncated normal distribution.

    Note:
    The function adjusts the bounds to the standard normal distribution's
    z-scores before creating the truncated distribution. The returned values
    are normalized so that they sum to 1.
    """

    # Convert the bounds to z-scores
    lower_bound_z = (lower_bound - mean) / sd
    upper_bound_z = (upper_bound - mean) / sd

    # Create the continous truncated normal distribution
    contin_dist = truncnorm(lower_bound_z, upper_bound_z, loc=mean, scale=sd)

    # Get all prob values at int steps
    dist_int_values = contin_dist.pdf(range(lower_bound, upper_bound))

    # Normalize values so they sum to 1.
    dist_int_values /= np.sum(dist_int_values)

    return dist_int_values


# Generate data
ntrials = 101
mean = 0
sigma = 1
confidence = np.array(random.sample(range(0, 101), ntrials))# generate random numbers
model_pred = np.full(ntrials, mean) #np.array(random.sample(range(0, 100), ntrials)) # generate random numbers
sigma_vec = np.full(ntrials, sigma)

# Small value to avoid division by 0
epsilon = 1e-10

# Initialize an empty array for NLLs
nlls = np.zeros(len(confidence))

start_time = time.perf_counter()  # Start the timer

# Loop over each trial and compute the NLL for each confidence value
for i in range(len(confidence)):
    # Create truncated normal distribution for each trial
    dist_values = create_truncated_normal_new(model_pred[i], sigma_vec[i],
                                              lower_bound=0, upper_bound=101)

    # Get the likelihood (NLL) for the current confidence value
    nlls[i] = dist_values[confidence[i]]

end_time = time.perf_counter()  # End the timer
execution_time = end_time - start_time  # Calculate elapsed time

# Print the NLLs and their sum
print("Sum of NLLs:", np.sum(nlls))
print(f"Execution time: {execution_time:.6f} seconds")

# Plot the confidence values against the NLLs
plt.scatter(confidence, nlls)
plt.xlabel("Confidence")
plt.ylabel("NLLs")
plt.title("Normalized Truncated Normal Distribution for Confidence Values")
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# This function creates a normal distrobution and then normalises it based on
# the discrete integers within the bounds so that the probabilities for these
# sum to one. The neat thing is that this is done using numpy broadcasting so
# to avoid having to use loops. This function is hence solving both earlier
# issues by producing correct probabilities for a specified range and does so
# faster than the earlier functions that normalised the curve.

# this is the function used in the model fitting scripts.

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

# Example data
confidence = np.array([0, 50, 55])
mean = np.array([0, 100, 60])
sigma = np.array([1, 15, 5])

# Calculate normalized probabilities and the full distributions
start_time = time.perf_counter()  # Start the timer
probabilities, y_values = calc_norm_prob_vectorized(confidence, mean, sigma)
end_time = time.perf_counter()  # End the timer
execution_time = end_time - start_time  # Calculate elapsed time

# Print probabilities and their sum
print("Probabilities:", probabilities)
print("Sum of probabilities (should not sum to 1, but the sum of each curve is 1):", np.sum(probabilities))

# Print time
print(f"Execution time: {execution_time:.6f} seconds")

# Plot the normalized normal distributions and the confidence points
x_values = np.arange(0, 101)  # Integer steps for plotting distributions

plt.figure(figsize=(10, 6))
for i in range(len(mean)):
    # Plot the normalized distribution for each mean and sigma
    plt.plot(x_values, y_values[i], label=f"Mean={mean[i]}, Sigma={sigma[i]}")
    print(sum(y_values[i]))
    # Mark the confidence value
    plt.scatter(confidence[i], y_values[i, confidence[i]], color='red')

# Add labels and title
plt.title("Normalized Normal Distributions with Confidence Values (Vectorized)")
plt.xlabel("Value")
plt.ylabel("Normalized Probability Density")
plt.legend()
plt.show()

#%% This part shows that this the last function also can be inverted.

def calc_inverted_norm_prob_vectorized(confidence, mean, sigma,
                                       lower_bound=0, upper_bound=100):
    """
    Calculate the inverted and normalized probability of confidence values given
    a normal distribution with corresponding mean and sigma values for each point.
    Ensure that the probability density values for integer steps between lower_bound
    and upper_bound sum to 1 for each distribution, using vectorized operations.

    Parameters:
    confidence (array): List or array of confidence values.
    mean (array): List or array of mean values corresponding to each confidence value.
    sigma (array): List or array of standard deviation values corresponding to each confidence value.
    lower_bound (int): The lower bound for the range of values to normalize over (default 0).
    upper_bound (int): The upper bound for the range of values to normalize over (default 100).

    Returns:
    numpy.ndarray: An array of inverted and normalized probabilities for each confidence[i]
                   based on the inverted and normalized normal distribution defined by mean[i]
                   and sigma[i].
    """

    # Create an array of integer steps [lower_bound, upper_bound] for all distributions
    x_values = np.arange(lower_bound, upper_bound + 1)

    # Reshape mean and sigma to broadcast across x_values (for element-wise operations)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]

    # Compute the Gaussian probability for each x_value, mean, and sigma (vectorized)
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)

    # Invert the distribution by subtracting from the maximum value for each distribution
    y_values = np.max(y_values, axis=1, keepdims=True) - y_values

    # Normalize the inverted distribution so that the sum of the values equals 1
    y_values /= np.sum(y_values, axis=1, keepdims=True)

    # Extract probabilities for the confidence values (vectorized)
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]

    return probabilities, y_values

# Example data
confidence = np.array([0, 50, 55])
mean = np.array([0, 100, 60])
sigma = np.array([1, 15, 5])

# Calculate normalized probabilities and the full distributions
start_time = time.perf_counter()  # Start the timer
probabilities, y_values = calc_inverted_norm_prob_vectorized(confidence, mean, sigma)

end_time = time.perf_counter()  # End the timer
execution_time = end_time - start_time  # Calculate elapsed time

# Print probabilities and their sum
print("Probabilities:", probabilities)
print("Sum of probabilities (should not sum to 1, but the sum of each curve is 1):", np.sum(probabilities))

# Print time
print(f"Execution time: {execution_time:.6f} seconds")

# Plot the normalized normal distributions and the confidence points
x_values = np.arange(0, 101)  # Integer steps for plotting distributions

plt.figure(figsize=(10, 6))
for i in range(len(mean)):
    # Plot the normalized distribution for each mean and sigma
    plt.plot(x_values, y_values[i], label=f"Mean={mean[i]}, Sigma={sigma[i]}")
    print(sum(y_values[i]))
    # Mark the confidence value
    plt.scatter(confidence[i], y_values[i, confidence[i]], color='red')

# Add labels and title
plt.title("Normalized Normal Distributions with Confidence Values (Vectorized)")
plt.xlabel("Value")
plt.ylabel("Normalized Probability Density")
plt.legend()
plt.show()


#%% The function can be split into the previous calc_norm_prob_vectorized
# and a separate function for inverting the distribution
# (i.e., invert_normal_distribution)

import numpy as np
import matplotlib.pyplot as plt

def calc_inverted_norm_prob_vectorized(confidence, mean, sigma,
                                       lower_bound=0, upper_bound=100):
    # Same function as before, returns inverted and normalized y_values
    x_values = np.arange(lower_bound, upper_bound + 1)
    mean = mean[:, np.newaxis]
    sigma = sigma[:, np.newaxis]
    y_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mean) / sigma) ** 2)
    y_values = np.max(y_values, axis=1, keepdims=True) - y_values
    y_values /= np.sum(y_values, axis=1, keepdims=True)
    probabilities = y_values[np.arange(len(confidence)), confidence - lower_bound]
    return probabilities, y_values

def invert_normal_distribution(probabilities):
    max_prob = np.max(probabilities)
    inverted = max_prob - probabilities
    normalized_inverted = inverted / np.sum(inverted)
    return normalized_inverted

# Example data
confidence = np.array([55])
mean = np.array([60])
sigma = np.array([5])

# Calculate inverted and normalized probabilities and the full distributions
probabilities, y_values = calc_inverted_norm_prob_vectorized(confidence, mean, sigma)

# Now use the 'invert_normal_distribution' to invert y_values
probabilities_, y_values_ = calc_norm_prob_vectorized(confidence, mean, sigma)
inverted_y_values = np.array([invert_normal_distribution(y) for y in y_values_])

# Plot both curves for comparison
x_values = np.arange(0, 101)

plt.figure(figsize=(10, 6))
for i in range(len(mean)):
    print(sum(y_values[i]))
    plt.plot(x_values, y_values[i], label=f"Calc Inverted - Mean={mean[i]}")
    plt.plot(x_values, inverted_y_values[i], '--', label=f"Invert Func - Mean={mean[i]}")

plt.title("Comparison of Inverted and Normalized Distributions")
plt.xlabel("Value")
plt.ylabel("Inverted Normalized Probability Density")
plt.legend()
plt.show()


