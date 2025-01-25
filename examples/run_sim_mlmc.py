from examples.mlmc_lognormal import LognormalPDESample, LognormalPDEModel
from estimator.mlmc_estimator import MLMCAdaptiveEstimator
import numpy as np
import logging
import time
from utils.utils import addLoggingLevel, clear_fenics_cache

addLoggingLevel('TRACE', logging.DEBUG - 5)
clear_fenics_cache()

rng = np.random.default_rng(seed=200)

logging.basicConfig(level=logging.DEBUG, format='{levelname}: {message}', style='{')

sample = LognormalPDESample(rng=rng)
model = LognormalPDEModel()

algorithm = MLMCAdaptiveEstimator(sample, model, Lmin=1, Lmax=5, alpha=2, beta=4)

start_time = time.time()
algorithm.run(mse_tol=2e-6)
end_time = time.time()
estimator = algorithm.final_estimator

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Cost per level per sample: {estimator.cost_per_level_per_sample}')