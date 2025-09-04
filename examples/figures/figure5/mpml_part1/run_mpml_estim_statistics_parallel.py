from examples.mlmc_lognormal import LognormalPDESample
from examples.mpml_lognormal_minres import MPLognormalPDEModelMinres as model_cls
from estimator.mpml_estimator import MPMLAdaptiveEstimator as adapt_alg
import numpy as np
import logging
import time
from utils.utils import get_git_commit_hash, save_commit_hash, copy_settings_to_output, addLoggingLevel, clear_fenics_cache, export_conda_environment
from concurrent.futures import ProcessPoolExecutor
import functools

addLoggingLevel('TRACE', logging.DEBUG - 5)

# Set the random seed for reproducibility
random_seed = 1692215 #Change for different method testing!
logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')
# Initialize the sample and model
model = model_cls()
Lmin = 1
Lmax = 7
alpha = 2
beta = 4
alpha_tol = 1
beta_tol = 2
k_p = 0.1

# Initialize the MLMC algorithm
mse_tol_array = [2e-6]
num_runs_per_tolerance = 1000
num_workers = 70

# Number of times to run the simulation
output_folder = "data/mpml_minres_kp1/"
this_file = "run_mpml_estim_statistics_parallel.py"
settings_folder = "examples/"


git_commit = get_git_commit_hash()
copy_settings_to_output(output_folder=output_folder, settings_folder=settings_folder, main_file=this_file)
save_commit_hash(output_folder, git_commit)
export_conda_environment(output_folder=output_folder)


def run_simulation(run_id, mse_tol, model, Lmin, Lmax, alpha, beta, alpha_tol, beta_tol, k_p, mse_tol_id):
    """
    Function to run a single simulation. run_simulation will be executed in parallel.
    """
    rng = np.random.default_rng(seed=random_seed + run_id + mse_tol_id)
    sample = LognormalPDESample(rng=rng)
    try:
        logging.debug(f"Starting simulation run {run_id}")
        algorithm = adapt_alg(
            sample, model,
            Lmin=Lmin, Lmax=Lmax,
            alpha=alpha, beta=beta,
            alpha_tol=alpha_tol,
            beta_tol=beta_tol,
            k_p=k_p
        )
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

        return {
            "estimate": estimator.estimate,
            "nsamp_per_level": estimator.nsamp_per_level,
            "cost_per_level_per_sample": estimator.cost_per_level_per_sample,
            "runtime": runtime,
            "run_id": run_id
        }
    except Exception as e:
        logging.error(f"Error in simulation run {run_id}: {e}")
        return None


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
    clear_fenics_cache()

    # Parallelize the inner loop
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            functools.partial(run_simulation, mse_tol=mse_tol, model=model, 
                              Lmin=Lmin, Lmax=Lmax, alpha=alpha, beta=beta, alpha_tol=alpha_tol, beta_tol=beta_tol, k_p=k_p, mse_tol_id=mse_tol_array.index(mse_tol)),
            range(1, num_runs_per_tolerance + 1)
        ))

    # Process the results after parallel execution
    for result in results:
        if result is not None:
            estimates.append(result["estimate"])
            nsamp_per_level_list.append(result["nsamp_per_level"])
            cost_per_level_per_sample_list.append(result["cost_per_level_per_sample"])
            runtimes.append(result["runtime"])

    # Save the results
    np.savez(
        simulation_data_file,
        estimates=np.array(estimates),
        nsamp_per_level=np.array(nsamp_per_level_list, dtype=object),  # Use dtype=object for variable-length arrays
        cost_per_level_per_sample=np.array(cost_per_level_per_sample_list, dtype=object),
        runtimes=np.array(runtimes)
    )
    logging.debug(f"Results saved to {simulation_data_file}.\n")

    end_time_tol = time.time()
    logging.info(f"Total runtime for tolerance {mse_tol}: {end_time_tol - start_time_tol}\n")

logging.info(f"All simulations completed and final results saved to {simulation_data_file}.")
