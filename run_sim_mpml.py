from examples.mpml_lognormal import MPLognormalPDEModel
from examples.mlmc_lognormal import LognormalPDESample
from estimator.mpml_estimator import MPMLAdaptiveEstimator
import numpy as np
import logging
import time

np.random.seed(17)
logging.basicConfig(level=logging.INFO, format='{levelname}: {message}', style='{')

sample = LognormalPDESample()
model = MPLognormalPDEModel()

estimator = MPMLAdaptiveEstimator(sample, model, Lmin=0, Lmax=5, alpha=2, beta=4, approximate_gamma=3, k_p=0.1, alpha_tol=1, beta_tol=2)

start_time = time.time()
estimator.run(mse_tol=8e-7)
end_time = time.time()

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Max level used: {estimator.max_level_used}')
logging.info(f'Computational error tolerance per level: {estimator.final_estimator.comp_tol_per_level}')