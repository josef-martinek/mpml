from setup.mlmc_lognormal import LognormalPDESample
from setup.mpml_lognormal_minres import LognormalMinresMonitor as model_cls
import numpy as np
import logging
import time
from utils.utils import get_git_commit_hash, save_commit_hash, copy_settings_to_output, addLoggingLevel, clear_fenics_cache, export_conda_environment
from concurrent.futures import ProcessPoolExecutor
import functools
import gc
import traceback

addLoggingLevel('TRACE', logging.DEBUG - 5)

# Set the random seed for reproducibility
random_seed = 207
logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')
# Initialize the sample and model
model = model_cls()
Lmax = 6

num_runs_per_level = 100
num_workers = 70

# Number of times to run the simulation
output_folder = "data/decay_plot7/"
this_file = "gen_data_decay_plot.py"
settings_folder = "setup/"


git_commit = get_git_commit_hash()
copy_settings_to_output(output_folder=output_folder, settings_folder=settings_folder, main_file=this_file)
save_commit_hash(output_folder, git_commit)
export_conda_environment(output_folder=output_folder)


def run_simulation(run_id, l):
    """
    Function to run a single simulation. run_simulation will be executed in parallel.
    """
    rng = np.random.default_rng(seed=random_seed + run_id + l)
    sample = LognormalPDESample(rng=rng)
    omega = sample.draw()
    
    try:
        #logging.debug(f"Starting simulation run {run_id} at level {l}")
        start_time = time.time()

        # Evaluate monitor at level l
        data_l = model.evaluate_monitor(level=l, sample=omega)
        qois_l = data_l["qois_per_iter"]
        residuals_l = data_l["rel_residuals"]

        # Evaluate monitor at level l-1
        data_lm1 = model.evaluate_monitor(level=l-1, sample=omega)
        qois_lm1 = data_lm1["qois_per_iter"]
        residuals_lm1 = data_lm1["rel_residuals"]

        end_time = time.time()
        runtime = end_time - start_time
        logging.debug(f"Total runtime for run {run_id} at levels {l} and {l-1}: {runtime}")

        gc.collect()

        return {
            "run_id": run_id,
            "omega": omega,
            "qois_l": qois_l,
            "qois_lm1": qois_lm1,
            "residuals_l": residuals_l,
            "residuals_lm1": residuals_lm1,
            "runtime": runtime,
            "qois_l_level": l,
            "qois_lm1_level": l-1,
        }
    except Exception as e:
        logging.error(f"Error in simulation run {run_id} at levels {l} and {l-1}:\n{traceback.format_exc()}")
        return None


# Loop over the number of runs
for l in np.arange(1, Lmax+1):
    # Initialize lists to collect results
    # Collect data arrays
    all_runtimes = []
    all_run_ids = []
    all_omegas = []
    all_qois_l = []
    all_qois_lm1 = []
    all_qois_l_levels = []
    all_qois_lm1_levels = []
    all_residuals_l = []
    all_residuals_lm1 = []
    simulation_data_file = output_folder + "level" + str(l) + "and" + str(l-1) + ".npz"
    logging.info(f"Starting simulation levels {l} and {l-1}")
    start_time_tol = time.time()
    clear_fenics_cache()

    # Parallelize the inner loop
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(
            functools.partial(run_simulation, l=l),
            range(1, num_runs_per_level + 1)
        ))

    # Process the results after parallel execution
    for result in results:
        if result is not None:
            all_run_ids.append(result["run_id"])
            all_runtimes.append(result["runtime"])
            all_omegas.append(result["omega"])
            all_qois_l.append(result["qois_l"])
            all_qois_lm1.append(result["qois_lm1"])
            all_qois_l_levels.append(result["qois_l_level"])
            all_qois_lm1_levels.append(result["qois_lm1_level"])
            all_residuals_l.append(result["residuals_l"])
            all_residuals_lm1.append(result["residuals_lm1"])

    # Save the results
    np.savez(
        simulation_data_file,
        run_ids=np.array(all_run_ids),
        omegas=np.array(all_omegas, dtype=object),
        runtimes=np.array(all_runtimes),
        qois_l=np.array(all_qois_l, dtype=object),
        qois_lm1=np.array(all_qois_lm1, dtype=object),
        residuals_l=np.array(all_residuals_l, dtype=object),
        residuals_lm1=np.array(all_residuals_lm1, dtype=object),
        qois_lm1_levels=np.array(all_qois_lm1_levels, dtype=object),
        qois_l_levels=np.array(all_qois_l_levels, dtype=object)
    )
    logging.debug(f"Results saved to {simulation_data_file}.\n")

    end_time_tol = time.time()
    logging.info(f"Total runtime for level {l}: {end_time_tol - start_time_tol}\n")

logging.info(f"All simulations completed and final results saved to {simulation_data_file}.")
