import os
import logging
import numpy as np
from utils.plotting_utils import load_simulation_data, plot_samples_single_tolerance


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{levelname}: {message}", style="{")

    # Specify tolerances (update as needed)
    tolerances = [1e-6]

    # Specify folders with results for method 1 and method 2
    method1_folder = "plots/mlmc_cholesky"
    method2_folder = "plots/mpml_itref"

    # Labels for the methods
    method1_label = "mlmc cholesky"
    method2_label = "mpml itref"

    # Initialize lists to store number of samples per level for each method
    method1_samples = []
    method2_samples = []

    # Loop over tolerances and load data for both methods
    for tol in tolerances:
        method1_file = os.path.join(method1_folder, f"mlmc_results_test{tol}.npz")
        method2_file = os.path.join(method2_folder, f"mlmc_results_test{tol}.npz")

        try:
            # Load data for method 1
            method1_data = load_simulation_data(method1_file)
            method1_nsamp = method1_data["nsamp_per_level"]
            method1_samples.append(method1_nsamp)
        except Exception as e:
            logging.error(f"Failed to process file {method1_file}: {e}")
            method1_samples.append(None)

        try:
            # Load data for method 2
            method2_data = load_simulation_data(method2_file)
            method2_nsamp = method2_data["nsamp_per_level"]
            method2_samples.append(method2_nsamp)
        except Exception as e:
            logging.error(f"Failed to process file {method2_file}: {e}")
            method2_samples.append(None)

    # Plot average number of samples with confidence intervals
    plot_samples_single_tolerance(method1_nsamp, method2_nsamp, method1_label, method2_label)