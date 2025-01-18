from examples.mpml_lognormal_itref import MPLognormalPDEModelItref
from examples.mlmc_lognormal import LognormalPDESample
from estimator.mpml_estimator import MPMLAdaptiveEstimator
import numpy as np
import logging
import time
from utils.utils import addLoggingLevel, clear_fenics_cache

addLoggingLevel('TRACE', logging.DEBUG - 5)
clear_fenics_cache()

rng = np.random.default_rng(seed=20)
logging.basicConfig(level=logging.TRACE, format='{levelname}: {message}', style='{')

sample = LognormalPDESample(rng=rng)
model = MPLognormalPDEModelItref()

algorithm = MPMLAdaptiveEstimator(sample, model, Lmin=1, Lmax=5, alpha=2, beta=4, k_p=0.1, alpha_tol=1, beta_tol=2)

start_time = time.time()
algorithm.run(mse_tol=8e-6)
end_time = time.time()
estimator = algorithm.final_estimator

logging.info(f'Total runtime: {end_time - start_time}')
logging.info(f'Estimated QOI: {estimator.estimate}')
logging.info(f'Final numer of samples per level: {estimator.nsamp_per_level}')
logging.info(f'Computational error tolerance per level: {estimator.final_estimator.comp_tol_per_level}')