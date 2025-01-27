import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from scipy.stats import t, sem


def load_simulation_data(file_path):
    """
    Loads simulation data from the specified file.

    Parameters:
    file_path (str): Path to the simulation data file (.npz).

    Returns:
    dict: A dictionary containing the loaded data (estimates, nsamp_per_level, cost_per_level_per_sample, runtimes).
    
    Raises:
    RuntimeError: If the file does not exist or cannot be loaded.
    """
    if not os.path.exists(file_path):
        raise RuntimeError(f"File '{file_path}' does not exist.")

    try:
        # Load the .npz file
        data = np.load(file_path, allow_pickle=True)

        # Extract the data
        estimates = data.get("estimates")
        nsamp_per_level = data.get("nsamp_per_level")
        cost_per_level_per_sample = data.get("cost_per_level_per_sample")
        runtimes = data.get("runtimes")

        logging.info(f"Successfully loaded data from {file_path}")
        return {
            "estimates": estimates,
            "nsamp_per_level": nsamp_per_level,
            "cost_per_level_per_sample": cost_per_level_per_sample,
            "runtimes": runtimes
        }
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading data from '{file_path}': {e}")
    
def compute_mean_values(data):
    """
    Computes the mean number of samples per level and mean cost per level across all runs.

    Parameters:
    data (dict): Dictionary containing simulation data with keys "nsamp_per_level" and "cost_per_level_per_sample".

    Returns:
    tuple: A tuple containing two numpy arrays:
           - mean_nsamp_per_level: Mean number of samples per level.
           - mean_cost_per_level: Mean cost per level.
    """
    nsamp_per_level = data["nsamp_per_level"]
    cost_per_level_per_sample = data["cost_per_level_per_sample"]

    # Determine the maximum number of levels across all runs
    max_levels = max(len(nsamp) for nsamp in nsamp_per_level)

    # Initialize arrays to accumulate sums for mean calculation
    total_nsamp = np.zeros(max_levels)
    total_cost = np.zeros(max_levels)
    num_runs = len(nsamp_per_level)

    for i in range(num_runs):
        # Extend shorter arrays with zeros to match max_levels
        nsamp = np.pad(nsamp_per_level[i], (0, max_levels - len(nsamp_per_level[i])), constant_values=0)
        cost = np.pad(cost_per_level_per_sample[i], (0, max_levels - len(cost_per_level_per_sample[i])), constant_values=0)

        # Accumulate the totals
        total_nsamp += nsamp
        total_cost += cost

    # Compute the means
    mean_nsamp_per_level = total_nsamp / num_runs
    mean_cost_per_level = total_cost / num_runs

    return mean_nsamp_per_level, mean_cost_per_level


def plot_mean_total_cost(simulations_data, simulation_labels):
    """
    Plots the mean total cost per level for the given simulations.

    Parameters:
    simulations_data (list of dict): List of simulation data dictionaries.
    simulation_labels (list of str): List of labels for each simulation.
    """
    plt.figure(figsize=(10, 6))

    for data, label in zip(simulations_data, simulation_labels):
        # Compute mean values
        mean_nsamp_per_level, mean_cost_per_level = compute_mean_values(data)

        # Compute mean total cost per level
        mean_total_cost_per_level = mean_nsamp_per_level * mean_cost_per_level

        # Plot the results
        levels = np.arange(len(mean_total_cost_per_level))
        plt.plot(levels[:-1], mean_total_cost_per_level[:-1], marker="o", label=label)

    # Customize the plot
    plt.xlabel("Level")
    plt.ylabel("Mean Total Cost Per Level")
    plt.yscale("log")
    plt.title("Mean Total Cost Per Level for Simulations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_total_mean_cost_vs_tolerance(tolerances, method1_costs, method2_costs, method1_label, method2_label):
    """
    Plots total mean cost vs MSE tolerance for two methods.

    Parameters:
    tolerances (list of float): List of MSE tolerances.
    method1_costs (list of float): List of total mean costs for method 1.
    method2_costs (list of float): List of total mean costs for method 2.
    method1_label (str): Label for method 1.
    method2_label (str): Label for method 2.
    """
    # Define consistent style settings
    fontsize = 16
    markersize = 16
    linewidth = 1.5

    # Adjusted figure
    plt.figure(figsize=(8, 6))

    # Plot method 1
    plt.loglog(tolerances, method1_costs, marker="v", linestyle="--", label=method1_label, 
               markersize=markersize, linewidth=linewidth)
    
    # Plot method 2
    plt.loglog(tolerances, method2_costs, marker="o", linestyle="--", label=method2_label, 
               markersize=markersize, linewidth=linewidth)

    # Adjust labels, title, and legend
    plt.xlabel("MSE Tolerance", fontsize=fontsize)
    #plt.ylabel("Total Mean Cost", fontsize=fontsize)
    #plt.title("Total Mean Cost vs MSE Tolerance", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc="upper right")

    # Adjust ticks and grid
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, which="both", linestyle="--")

    # Adjust layout and save as PDF
    plt.tight_layout()
    plt.savefig("total_mean_cost_vs_tolerance.pdf", format="pdf", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()


def compute_mean_squared_error(estimates, reference):
    """
    Computes the mean squared error (MSE) of the estimates with respect to a reference value.

    Parameters:
    estimates (list or ndarray): Estimates from the simulation.
    reference (float): Reference value for comparison.

    Returns:
    float: Mean squared error.
    """
    return np.mean((np.array(estimates) - reference) ** 2)


def plot_mse_vs_tolerance(tolerances, method1_mse, method2_mse, method1_label, method2_label):
    """
    Plots the mean squared error (MSE) vs tolerances for two methods.

    Parameters:
    tolerances (list of float): List of MSE tolerances.
    method1_mse (list of float): List of MSE values for method 1.
    method2_mse (list of float): List of MSE values for method 2.
    method1_label (str): Label for method 1.
    method2_label (str): Label for method 2.
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(tolerances, method1_mse, marker="o", label=method1_label)
    plt.loglog(tolerances, method2_mse, marker="s", label=method2_label)
    plt.loglog(tolerances, tolerances, marker="x", label="mse tol")

    plt.xlabel("MSE Tolerance")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Mean Squared Error vs Tolerances")
    plt.legend()
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()

    # Show the plot
    plt.show()

def compute_mse_and_confidence_interval(estimates, reference, confidence=0.95):
    """
    Computes the mean squared error (MSE) and its confidence interval.

    Parameters:
    estimates (list or ndarray): Estimates from the simulation.
    reference (float): Reference value for comparison.
    confidence (float): Confidence level (default is 0.95 for 95%).

    Returns:
    tuple: (mse, lower_bound, upper_bound)
    """
    estimates = np.array(estimates)
    n = len(estimates)
    mse = np.mean((estimates - reference) ** 2)

    # Compute standard error of the MSE
    variance = np.var((estimates - reference) ** 2, ddof=1)  # Variance of squared errors
    std_error = np.sqrt(variance / n)

    # Critical value for t-distribution
    t_critical = t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * std_error

    lower_bound = mse - margin_of_error
    upper_bound = mse + margin_of_error

    return mse, lower_bound, upper_bound


def plot_mse_with_confidence_intervals(tolerances, method1_results, method2_results, method1_label, method2_label):
    """
    Plots the mean squared error (MSE) with 95% confidence intervals vs tolerances for two methods.

    Parameters:
    tolerances (list of float): List of tolerances.
    method1_results (list of tuple): List of (mse, lower_bound, upper_bound) for method 1.
    method2_results (list of tuple): List of (mse, lower_bound, upper_bound) for method 2.
    method1_label (str): Label for method 1.
    method2_label (str): Label for method 2.
    """
    # Define consistent style settings
    fontsize = 16
    markersize = 16
    linewidth = 1.5

    # Adjusted figure
    plt.figure(figsize=(8, 6))

    # Unpack MSE and confidence intervals for plotting
    method1_mse, method1_lower, method1_upper = zip(*method1_results)
    method2_mse, method2_lower, method2_upper = zip(*method2_results)

    # Plot method 1
    plt.errorbar(tolerances, method1_mse,
                yerr=[np.array(method1_mse) - np.array(method1_lower),
                    np.array(method1_upper) - np.array(method1_mse)],
                fmt="v--", label=method1_label, capsize=5, markersize=markersize, linewidth=linewidth)

    # Plot method 2
    plt.errorbar(tolerances, method2_mse,
                yerr=[np.array(method2_mse) - np.array(method2_lower),
                    np.array(method2_upper) - np.array(method2_mse)],
                fmt="o--", label=method2_label, capsize=5, markersize=markersize, linewidth=linewidth)

    # Add mse tolerance line
    plt.loglog(tolerances, tolerances, marker="x", label="MSE Tolerance", linewidth=linewidth, markersize=markersize)

    # Adjust labels, title, and legend
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("MSE Tolerance", fontsize=fontsize)
    #plt.ylabel("Mean Squared Error (MSE)", fontsize=fontsize)
    #plt.title("Mean Squared Error vs Tolerances with 95% Confidence Intervals", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc="upper left")

    # Adjust ticks and grid
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, which="both", linestyle="--")

    # Adjust layout and save as PDF
    plt.tight_layout()
    plt.savefig("mse_with_confidence_intervals.pdf", format="pdf", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()

def compute_confidence_interval(data_list, confidence=0.95):
    """
    Computes element-wise averages and 95% confidence intervals for a list of numpy arrays.

    Parameters:
    data_list (list of ndarray): List of numpy arrays containing the number of samples.
    confidence (float): Confidence level, default is 0.95 for a 95% confidence interval.

    Returns:
    tuple: (mean_array, lower_bound_array, upper_bound_array)
           Each is a numpy array containing the element-wise results.
    """
    # Find the length of the longest array
    max_length = max(len(arr) for arr in data_list)
    
    # Pad all arrays to the same length with zeros
    padded_data = np.array([np.pad(arr, (0, max_length - len(arr)), constant_values=0) for arr in data_list])
    
    # Compute the element-wise mean
    mean_array = np.mean(padded_data, axis=0)
    
    # Compute the standard error of the mean (SEM) element-wise
    stderr_array = sem(padded_data, axis=0)
    
    # Degrees of freedom for t-distribution
    n = padded_data.shape[0]
    if n < 2:  # Can't compute confidence interval with fewer than 2 arrays
        return mean_array, mean_array, mean_array

    # Compute the t critical value
    t_crit = t.ppf((1 + confidence) / 2, df=n - 1)
    
    # Compute the margin of error element-wise
    margin_of_error = t_crit * stderr_array
    
    # Compute the confidence interval bounds element-wise
    lower_bound_array = mean_array - margin_of_error
    upper_bound_array = mean_array + margin_of_error
    
    return mean_array, lower_bound_array, upper_bound_array


def plot_samples_single_tolerance(method1_samples, method2_samples, method1_label, method2_label):
    """
    Plots the average number of samples per level with 95% confidence intervals for two methods
    for a single error tolerance.

    Parameters:
    method1_samples (list of ndarray): Number of samples per level for Method 1.
    method2_samples (list of ndarray): Number of samples per level for Method 2.
    method1_label (str): Label for Method 1.
    method2_label (str): Label for Method 2.
    """
    # Compute levels based on the lengths of sample arrays
    levels = np.arange(min(
        max(len(nsamp) for nsamp in method1_samples),
        max(len(nsamp) for nsamp in method2_samples)
    ))

    # Compute confidence intervals for both methods
    method1_means, method1_lowers, method1_uppers = compute_confidence_interval(method1_samples)
    method2_means, method2_lowers, method2_uppers = compute_confidence_interval(method2_samples)

    # Plot settings
    fontsize = 16
    markersize = 12
    linewidth = 1.5

    # Create the figure
    plt.figure(figsize=(8, 6))

    # Plot Method 1 with error bars
    plt.errorbar(
        levels,
        method1_means[levels],
        yerr=[
            np.array(method1_means[levels]) - np.array(method1_lowers[levels]),
            np.array(method1_uppers[levels]) - np.array(method1_means[levels]),
        ],
        fmt="o--",
        label=method1_label,
        capsize=5,
        color="blue",
        markersize=markersize,
        linewidth=linewidth,
    )

    # Plot Method 2 with error bars
    plt.errorbar(
        levels,
        method2_means[levels],
        yerr=[
            np.array(method2_means[levels]) - np.array(method2_lowers[levels]),
            np.array(method2_uppers[levels]) - np.array(method2_means[levels]),
        ],
        fmt="v--",
        label=method2_label,
        capsize=5,
        color="red",
        markersize=markersize,
        linewidth=linewidth,
    )

    # Set x-axis and y-axis scale
    plt.xscale("linear")
    plt.yscale("log")

    # Axis labels and title
    plt.xlabel("Level", fontsize=fontsize)
    plt.ylabel("Average Number of Samples", fontsize=fontsize)
    #plt.title("Average Number of Samples per Level with 95% Confidence Interval", fontsize=fontsize)

    # Legend and grid
    plt.legend(fontsize=fontsize, loc="best")
    plt.grid(True, linestyle="--", which="both")

    # Ticks
    plt.xticks(levels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Adjust layout and save as PDF
    plt.tight_layout()
    plt.savefig("average_samples_per_level.pdf", format="pdf", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()



def plot_mse_with_confidence_intervals_var_kp(method1_results, method1_label):
    """
    Plots the mean squared error (MSE) with 95% confidence intervals vs k_p values.

    Parameters:
    method1_results (list of tuple): List of (mse, lower_bound, upper_bound) for method 1.
    method1_label (str): Label for method 1.
    """
    # Define consistent style settings
    fontsize = 16
    markersize = 16
    linewidth = 1.5

    # Adjusted figure
    plt.figure(figsize=(8, 6))

    # Unpack MSE and confidence intervals for plotting
    method1_mse, method1_lower, method1_upper = zip(*method1_results)
    kp = [0.05, 0.1, 0.2, 0.4]

    # Plot method 1 with error bars
    plt.errorbar(
        kp,
        method1_mse,
        yerr=[
            np.array(method1_mse) - np.array(method1_lower),
            np.array(method1_upper) - np.array(method1_mse),
        ],
        fmt="o",
        label=method1_label,
        capsize=5,
        markersize=markersize,
        linewidth=linewidth,
    )

    # Add a horizontal line for MLMC minres mse
    plt.axhline(
        y=3.0870759359968266e-06,
        color="black",
        linestyle="--",
        linewidth=linewidth,
        label=f"MLMC MINRES MSE",
    )

    # Add semi-transparent error region around the constant line
    xmin = 0.04
    xmax = 0.45
    plt.fill_between(
        [xmin, xmax],
        [2.330957660115031e-06, 2.330957660115031e-06],
        [3.843194211878622e-06, 3.843194211878622e-06],
        color="grey",
        alpha=0.2,
    )

    # Log scales for both axes
    plt.xscale("log")
    plt.yscale("log")

    # Labels, title, legend
    plt.xlabel("$k_p$ value", fontsize=fontsize)
    #plt.ylabel("Mean Squared Error (MSE)", fontsize=fontsize)
    #plt.title("Mean Squared Error vs k_p Values with 95% Confidence Intervals", fontsize=fontsize)
    plt.legend(fontsize=fontsize, loc="upper right")

    # Ticks, grid, and axis limits
    plt.xticks(kp, labels=kp, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(True, which="both", linestyle="--")
    plt.xlim(xmin, xmax)

    # Adjust layout and save as PDF
    plt.tight_layout()
    plt.savefig("mse_vs_kp_with_intervals.pdf", format="pdf", bbox_inches="tight", dpi=300)

    # Show the plot
    plt.show()

