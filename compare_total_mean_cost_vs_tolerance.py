import logging
import os
import numpy as np
from utils.plotting_utils import load_simulation_data, plot_total_mean_cost_vs_tolerance, compute_mean_values


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{levelname}: {message}", style="{")

    # Specify tolerances (update as needed)
    tolerances = [8e-6, 4e-6, 2e-6, 1e-6]

    # Specify folders with results for method 1 and method 2
    #method1_folder = "plots/mlmc_minres"
    method1_folder = "plots/mlmc_lu"
    #method2_folder = "plots/mpml_minres"
    method2_folder = "plots/mpml_itref"

    # Labels for the methods
    #method1_label = "mlmc minres"
    method1_label = "mlmc lu"
    #method2_label = "mpml minres"
    method2_label = "mpml itref"

    # Initialize lists to store total mean costs for each method
    method1_costs = []
    method2_costs = []

    # Loop over tolerances and load data for both methods
    for tol in tolerances:
        method1_file = os.path.join(method1_folder, f"mlmc_results_test{tol}.npz")
        method2_file = os.path.join(method2_folder, f"mlmc_results_test{tol}.npz")

        try:
            # Load data for method 1
            method1_data = load_simulation_data(method1_file)
            mean_nsamp_per_level_1, mean_cost_per_level_1 = compute_mean_values(method1_data)
            method1_cost = np.sum(mean_nsamp_per_level_1*mean_cost_per_level_1)
            method1_costs.append(method1_cost)
        except Exception as e:
            logging.error(f"Failed to process file {method1_file}: {e}")
            method1_costs.append(None)

        try:
            # Load data for method 2
            method2_data = load_simulation_data(method2_file)
            mean_nsamp_per_level_2, mean_cost_per_level_2 = compute_mean_values(method2_data)
            method2_cost = np.sum(mean_nsamp_per_level_2*mean_cost_per_level_2)
            method2_costs.append(method2_cost)
        except Exception as e:
            logging.error(f"Failed to process file {method2_file}: {e}")
            method2_costs.append(None)

    # Plot total mean cost vs MSE tolerance
    plot_total_mean_cost_vs_tolerance(tolerances, method1_costs, method2_costs, method1_label, method2_label)