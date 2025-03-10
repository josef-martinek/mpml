"""
This script runs the adaptive Mixed Precision MLMC algorithm for estimating
expectation of a quantity of interest.
The problem is given by an elliptic PDE with lognormal random coefficient, the
quantity of interest is given by the integral over the domain.
Low precision Cholesky factorisation with iterative refinement is used as the
linear solver. Iterative refinement setup is specified in the class
MPLognormalPDEModelItrefQP. Stopping criterion for iterative refinement is chosen
adaptively.

"""

from setup.mpml_lognormal_itref_qp import LognormalPDESampleQP
from setup.mpml_lognormal_itref_qp import MPLognormalPDEModelItrefQP
from core.estimator.mpml_estimator import MPMLAdaptiveEstimator
import numpy as np
import logging
import time
from utils.utils import addLoggingLevel, clear_fenics_cache

addLoggingLevel('TRACE', logging.DEBUG - 5)
clear_fenics_cache()

rng = np.random.default_rng(seed=15)
logging.basicConfig(level=logging.DEBUG, format='{levelname}: {message}', style='{')

sample = LognormalPDESampleQP(rng=rng)
model = MPLognormalPDEModelItrefQP()

algorithm = MPMLAdaptiveEstimator(sample, model, Lmin=0, Lmax=5, alpha=2, beta=4, k_p=0.05, alpha_tol=1, beta_tol=2)

start_time = time.time()
algorithm.run(mse_tol=8e-6, init_nsamp=1)
end_time = time.time()
estimator = algorithm.final_estimator

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Computational error tolerance per level: {estimator.comp_tol_per_level}')
logging.info(f'Cost per level per sample: {estimator.cost_per_level_per_sample}')
