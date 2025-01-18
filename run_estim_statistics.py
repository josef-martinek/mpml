from examples.mlmc_lognormal import LognormalPDESample, LognormalPDEModel
from estimator.mlmc_estimator import MLMCAdaptiveEstimator as adapt_alg
import numpy as np
import logging
import time
import os
from utils.utils import get_git_commit_hash, save_commit_hash, copy_settings_to_output, addLoggingLevel, clear_fenics_cache

addLoggingLevel('TRACE', logging.DEBUG - 5)

# Set the random seed for reproducibility
np.random.seed(26)
logging.basicConfig(level=logging.DEBUG, format='{levelname}: {message}', style='{')

# Initialize the sample and model
sample = LognormalPDESample()
model = LognormalPDEModel()
Lmin = 1
Lmax = 6
alpha = 2
beta = 4
approximate_gamma = 3

# Initialize the MLMC algorithm
mse_tol_array = [1e-7]
num_runs = 10

# Number of times to run the simulation
output_folder = "data/"
main_file = "run_estim_statistics.py"
settings_folder = "examples/"


git_commit = get_git_commit_hash()
copy_settings_to_output(output_folder=output_folder, settings_folder=settings_folder, main_file=main_file)
save_commit_hash(output_folder, git_commit)

# Loop over the number of runs
for mse_tol in mse_tol_array:
    # Initialize lists to collect results
    estimates = []
    nsamp_per_level_list = []
    cost_per_level_per_sample_list = []
    runtimes = []
    simulation_data_file = output_folder + "mlmc_results_test" + str(mse_tol) + ".npz"
    logging.info(f"Starting simulation tolerance {mse_tol}")
    start_time_tol = time.time()
    for run_id in range(1, num_runs + 1):
        clear_fenics_cache()
        logging.debug(f"Starting simulation run {run_id}")
        algorithm = adapt_alg(sample, model, Lmin=Lmin, Lmax=Lmax, alpha=alpha, beta=beta, approximate_gamma=approximate_gamma)
        # Measure runtime
        start_time = time.time()
        algorithm.run(mse_tol=mse_tol, init_nsamp=50)
        end_time = time.time()

        # Get the final estimator
        estimator = algorithm.final_estimator

        # Log the results
        runtime = end_time - start_time
        logging.debug(f"Total runtime for run {run_id}: {runtime}")
        logging.debug(f"Estimated QOI (run {run_id}): {estimator.estimate}")
        logging.info(f"Final number of samples per level (run {run_id}): {estimator.nsamp_per_level}")
        logging.debug(f"Cost per level per sample (run {run_id}): {estimator.cost_per_level_per_sample}")

        # Collect results for this run
        estimates.append(estimator.estimate)
        nsamp_per_level_list.append(estimator.nsamp_per_level)
        cost_per_level_per_sample_list.append(estimator.cost_per_level_per_sample)
        runtimes.append(runtime)

        # Save the intermediate results after every iteration
        np.savez(
            simulation_data_file,
            estimates=np.array(estimates),
            nsamp_per_level=np.array(nsamp_per_level_list, dtype=object),  # Use dtype=object for variable-length arrays
            cost_per_level_per_sample=np.array(cost_per_level_per_sample_list, dtype=object),
            runtimes=np.array(runtimes)
        )
        logging.debug(f"Checkpoint saved after run {run_id} to {simulation_data_file}.\n")
    end_time_tol = time.time()
    logging.info(f"Total runtime for tolerance {mse_tol}: {end_time_tol - start_time_tol}")

logging.info(f"All simulations completed and final results saved to {simulation_data_file}.")