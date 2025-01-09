from examples.mlmc_lognormal import LognormalPDESample, LognormalPDEModel
from estimator.mlmc_estimator import MLMCAdaptiveEstimator
import numpy as np
import logging

np.random.seed(15)
logging.basicConfig(level=logging.DEBUG, format='{levelname}: {message}', style='{')

sample = LognormalPDESample()
model = LognormalPDEModel()

estimator = MLMCAdaptiveEstimator(sample, model, Lmin=0, Lmax=5, alpha=2, beta=4, approximate_gamma=3)

estimator.run(mse_tol=8e-7)
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Max level used: {estimator.max_level_used}')