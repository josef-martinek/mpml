"""
This script runs the standard adaptive Multilevel Monte Carlo (MLMC) algorithm
for estimating expectation of a quantity of interest.
The problem is given by an elliptic PDE with lognormal random coefficient, the
quantity of interest is given by the integral over the domain.
PETSc MINRES with a fixed stopping criterion (given by the relative residual)
is used as the linear solver.

"""

from setup.mlmc_lognormal import LognormalPDESample
from setup.mlmc_lognormal_minres_fixtol import MLLognormalPDEModelMinres
from core.estimator.mlmc_estimator import MLMCAdaptiveEstimator
import numpy as np
import logging
import time
from utils.utils import addLoggingLevel, clear_fenics_cache

addLoggingLevel('TRACE', logging.DEBUG - 5)
clear_fenics_cache()

rng = np.random.default_rng(seed=200)

logging.basicConfig(level=logging.DEBUG, format='{levelname}: {message}', style='{')

sample = LognormalPDESample(rng=rng)
model = MLLognormalPDEModelMinres()

algorithm = MLMCAdaptiveEstimator(sample, model, Lmin=1, Lmax=5, alpha=2, beta=4)

start_time = time.time()
algorithm.run(mse_tol=2e-6)
end_time = time.time()
estimator = algorithm.final_estimator

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Cost per level per sample: {estimator.cost_per_level_per_sample}')