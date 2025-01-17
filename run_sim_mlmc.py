from examples.mlmc_lognormal import LognormalPDESample, LognormalPDEModel
from estimator.mlmc_estimator import MLMCAdaptiveEstimator
import numpy as np
import logging
import time

np.random.seed(22)
logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')

sample = LognormalPDESample()
model = LognormalPDEModel()

algorithm = MLMCAdaptiveEstimator(sample, model, Lmin=0, Lmax=5, alpha=2, beta=4, approximate_gamma=3)

start_time = time.time()
algorithm.run(mse_tol=8e-6)
end_time = time.time()
estimator = algorithm.final_estimator

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Cost per level per sample: {estimator.cost_per_level_per_sample}')