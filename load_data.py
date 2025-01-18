import numpy as np
import os
import logging

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


# Example usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="{levelname}: {message}", style="{")

    # Specify the file paths (update this with your simulation file paths)
    simulation_data_files = ["data/mlmc_results_test2e-06.npz"]

    for file_path in simulation_data_files:
        try:
            # Load the data
            data = load_simulation_data(file_path)
            a = None

        except Exception as e:
            logging.error(f"Failed to process file {file_path}: {e}")
