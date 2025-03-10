from examples.mpml_lognormal_itref_qp import LognormalPDESampleQP as sample_cls
from examples.mpml_lognormal_itref_qp import MPLognormalPDEModelItrefQP as model_cls
from estimator.mpml_estimator import MPMLNonAdaptiveEstimator as alg
from estimator.mpml_estimator import MPMLAdaptiveEstimator as adapt_alg
import numpy as np
import logging
import time
from utils.utils import get_git_commit_hash, save_commit_hash, copy_settings_to_output, addLoggingLevel, clear_fenics_cache, export_conda_environment
from concurrent.futures import ProcessPoolExecutor
import functools
import traceback

addLoggingLevel('TRACE', logging.DEBUG - 5)

# Set the random seed for reproducibility
random_seed = 4041 #Change for different method testing!!!
logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')
# Initialize the sample and model
model = model_cls()
Lmin = 0
Lmax = 6
alpha = 2
beta = 4
alpha_tol = 1
beta_tol = 2
k_p = 0.4

# Initialize the MLMC algorithm
num_samp_list = [np.array([5, 1, 1]), np.array([10, 2, 1]), np.array([19, 3, 1])]
num_runs_per_tolerance = 1000
num_workers = 72

# Number of times to run the simulation
output_folder = "data/itref_simple_fix_nsamp/"
this_file = "run_mpml_fix_nsamp_statistics.py"
settings_folder = "examples/"


git_commit = get_git_commit_hash()
copy_settings_to_output(output_folder=output_folder, settings_folder=settings_folder, main_file=this_file)
save_commit_hash(output_folder, git_commit)
export_conda_environment(output_folder=output_folder)


def run_simulation(run_id, num_samp, model, Lmin, num_samp_id):
    """
    Function to run a single simulation. run_simulation will be executed in parallel.
    """
    rng = np.random.default_rng(seed=random_seed + run_id + num_samp_id)
    sample = sample_cls(rng=rng)
    adapt_alg_inst = adapt_alg(sample, model, Lmin, Lmax, alpha, beta, alpha_tol, beta_tol, k_p)
    comp_tol = adapt_alg_inst._get_comp_tol(Lmin+len(num_samp)-1)
    try:
        logging.debug(f"Starting simulation run {run_id}")
        algorithm = alg(
            sample, model, nsamp_per_level=num_samp,
            Lmin=Lmin,
            comp_tol_per_level=comp_tol
        )
        # Measure runtime
        start_time = time.time()
        algorithm.run()
        end_time = time.time()

        # Log the results
        runtime = end_time - start_time
        logging.debug(f"Total runtime for run {run_id}: {runtime}")
        logging.info(f"Final estimate (run {run_id}): {algorithm.estimate}")
        logging.debug(f"Cost per level per sample (run {run_id}): {algorithm.cost_per_level_per_sample}")

        return {
            "estimate": algorithm.estimate,
            "nsamp_per_level": algorithm.nsamp_per_level,
            "cost_per_level_per_sample": algorithm.cost_per_level_per_sample,
            "runtime": runtime,
            "run_id": run_id
        }
    except Exception as e:
        logging.error(f"Error in simulation run {run_id}: {e}")
        traceback.print_exc()
        return None


# Loop over the number of tolerances
counter = 0
for num_samp in num_samp_list:
    # Initialize lists to collect results
    estimates = []
    nsamp_per_level_list = []
    cost_per_level_per_sample_list = []
    runtimes = []
    simulation_data_file = output_folder + "mlmc_results_test" + str(counter) + ".npz"
    logging.info(f"Starting simulation with number of samples {num_samp}")
    start_time_tol = time.time()
    clear_fenics_cache()

    # Parallelize the inner loop
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            functools.partial(run_simulation, num_samp=num_samp, model=model, 
                              Lmin=Lmin, num_samp_id=counter),
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
    logging.info(f"Total runtime for number of samples {num_samp}: {end_time_tol - start_time_tol}\n")
    counter += 1

logging.info(f"All simulations completed and final results saved to {simulation_data_file}.")
