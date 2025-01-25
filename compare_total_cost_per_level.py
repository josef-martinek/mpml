import logging
from utils.plotting_utils import load_simulation_data, plot_mean_total_cost


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{levelname}: {message}", style="{")

    # Specify the file paths (update these with your simulation file paths)
    simulation_data_files = [
        "data/mlmc_minres/mlmc_results_test1e-06.npz",
        "data/mpml_minres1/mlmc_results_test1e-06.npz"
    ]

    # Load the data
    simulations_data = []
    for file_path in simulation_data_files:
        try:
            data = load_simulation_data(file_path)
            simulations_data.append(data)
        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")

    # Define labels for the simulations
    simulation_labels = ["MLMC minres", "MPML minres"]

    # Plot the mean total cost per level
    plot_mean_total_cost(simulations_data, simulation_labels)
