import logging
import os
import numpy as np
from utils.plotting_utils import load_simulation_data, compute_mean_squared_error, plot_mse_vs_tolerance, compute_mse_and_confidence_interval, plot_mse_with_confidence_intervals


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{levelname}: {message}", style="{")

    # Specify tolerances (update as needed)
    tolerances = [8e-6, 4e-6, 2e-6, 1e-6]

    # Specify folders with results for method 1, method 2, and the reference solution
    method1_folder = "plots/mlmc_cholesky"
    method2_folder = "plots/mpml_itref"
    reference_folder = "data/reference_sol1"

    # Reference file (assuming it contains the reference solution)
    reference_file = os.path.join(reference_folder, "mlmc_results_test2e-08.npz")

    # Labels for the methods
    method1_label = "mlmc cholesky"
    method2_label = "mpml itref"

    # Load the reference solution
    try:
        reference_data = load_simulation_data(reference_file)
        reference_estimates = reference_data["estimates"]
        reference_mean = np.mean(reference_estimates)  # Compute the reference mean
    except Exception as e:
        logging.error(f"Failed to load reference solution: {e}")
        exit(1)

    # Initialize lists to store MSE values and confidence intervals for each method
    method1_results = []
    method2_results = []

    # Loop over tolerances and load data for both methods
    for tol in tolerances:
        method1_file = os.path.join(method1_folder, f"mlmc_results_test{tol}.npz")
        method2_file = os.path.join(method2_folder, f"mlmc_results_test{tol}.npz")

        try:
            # Load data for method 1
            method1_data = load_simulation_data(method1_file)
            method1_estimates = method1_data["estimates"]
            mse1, lower1, upper1 = compute_mse_and_confidence_interval(method1_estimates, reference_mean)
            method1_results.append((mse1, lower1, upper1))
        except Exception as e:
            logging.error(f"Failed to process file {method1_file}: {e}")
            method1_results.append((None, None, None))

        try:
            # Load data for method 2
            method2_data = load_simulation_data(method2_file)
            method2_estimates = method2_data["estimates"]
            mse2, lower2, upper2 = compute_mse_and_confidence_interval(method2_estimates, reference_mean)
            method2_results.append((mse2, lower2, upper2))
        except Exception as e:
            logging.error(f"Failed to process file {method2_file}: {e}")
            method2_results.append((None, None, None))

    # Plot MSE vs tolerances with confidence intervals
    plot_mse_with_confidence_intervals(tolerances, method1_results, method2_results, method1_label, method2_label)