from setup.mlmc_lognormal import LognormalPDESample, LognormalPDEModel
from core.estimator.mlmc_estimator import MLMCNonAdaptiveEstimator
import numpy as np
import logging
import time
from utils.utils import addLoggingLevel, clear_fenics_cache

addLoggingLevel('TRACE', logging.DEBUG - 5)
clear_fenics_cache()

rng = np.random.default_rng(seed=15)

logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')

sample = LognormalPDESample(rng=rng)
model = LognormalPDEModel()

num_samp = np.array([20, 5, 1])

algorithm = MLMCNonAdaptiveEstimator(sample, model, nsamp_per_level=num_samp, Lmin=1)

start_time = time.time()
algorithm.run()
end_time = time.time()

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {algorithm.estimate}')