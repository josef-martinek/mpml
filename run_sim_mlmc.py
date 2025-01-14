from examples.mlmc_lognormal import LognormalPDESample, LognormalPDEModel
from estimator.mlmc_estimator import MLMCAdaptiveEstimator
import numpy as np
import logging
import time

np.random.seed(17)
logging.basicConfig(level=logging.DEBUG, format='{levelname}: {message}', style='{')

sample = LognormalPDESample()
model = LognormalPDEModel()

estimator = MLMCAdaptiveEstimator(sample, model, Lmin=0, Lmax=5, alpha=2, beta=4, approximate_gamma=3)

start_time = time.time()
estimator.run(mse_tol=8e-7)
end_time = time.time()

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Max level used: {estimator.max_level_used}')