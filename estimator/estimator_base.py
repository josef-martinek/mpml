from model.model_base import ModelBase
from sample.sample_base import SampleBase
import numpy as np
from abc import ABC, abstractmethod
import logging
import time


class MLMCNonAdaptiveEstimatorBase(ABC):

    @property
    @abstractmethod
    def nsamp_per_level(self):
        pass

    @nsamp_per_level.setter
    @abstractmethod
    def nsamp_per_level(self, value):
        pass

    @property
    @abstractmethod
    def est_per_level(self):
        pass

    @property
    @abstractmethod
    def var_per_level(self):
        pass

    @property
    @abstractmethod
    def est_per_level_adjusted(self):
        pass

    @property
    @abstractmethod
    def var_per_level_adjusted(self):
        pass

    @property
    @abstractmethod
    def cost_per_level_per_sample(self):
        pass

    def run(self, *args, **kwargs):
        logging.debug(f'ML estimator nsamp: {self.nsamp_per_level}')
        start_time = time.time()
        self._run_impl(*args, **kwargs)
        end_time = time.time()
        logging.debug(f'ML estimator runtime: {end_time - start_time}')

    @abstractmethod
    def _run_impl(self):
        pass

    @abstractmethod
    def adjust_estimates_and_variances(self, alpha, beta):
        pass


class MLMCAdaptiveEstimatorBase(ABC):
    def __init__(self, sample: SampleBase, model: ModelBase, Lmin, Lmax, alpha, beta, approximate_gamma):
        self._sample = sample
        self._model = model
        if Lmin >= Lmax:
            raise ValueError("Estimator must have at least 2 levels.")
        self._Lmin = Lmin
        self._Lmax = Lmax
        self._alpha = alpha
        self._beta = beta
        self._gamma = approximate_gamma
        self._r = 0.7
        self._min_nsamp = 2
        self._reset_results()

    def _reset_results(self):
        self._final_estimator = None
        self._conv_success = None
        self._estimate = None
        self._nsamp_per_level = None
        self._max_level_used = None

    @property
    def conv_success(self):
        return self._conv_success

    @property
    def estimate(self):
        return self._estimate

    @property
    def nsamp_per_level(self):
        return self._nsamp_per_level

    @property
    def max_level_used(self):
        return self._max_level_used
    
    @property
    def final_estimator(self):
        return self._final_estimator

    def run(self, mse_tol, init_nsamp=10):
        self._reset_results()
        cur_max_level = self._Lmin + 1
        cur_ml_estimator = self._setup_nonadaptive_ml_estimator(init_nsamp, cur_max_level)
        while cur_max_level <= self._Lmax:
            cur_ml_estimator.run()
            cur_ml_estimator.adjust_estimates_and_variances(self._alpha, self._beta)
            new_max_level, conv_success = self._conv_check(cur_max_level, cur_ml_estimator, mse_tol)
            if conv_success:
                logging.info('Convergence successful')
                self._conv_success = True
                self._save_results(cur_ml_estimator, cur_max_level)
                return
            if new_max_level <= self._Lmax: 
                cur_ml_estimator = self._update_nonadaptive_ml_estimator(cur_ml_estimator, new_max_level, mse_tol, init_nsamp)
            cur_max_level = new_max_level
        self._conv_success = False
        if cur_max_level > self._Lmax:
            logging.warning('Convergence not achieve, max level exceeded.')
            self._save_results(cur_ml_estimator, cur_max_level)
        else:
            raise RuntimeError("Convergence not achieved, but max level not exceeded. That means bug.")

    @abstractmethod
    def _setup_nonadaptive_ml_estimator(nsamp, max_level) -> MLMCNonAdaptiveEstimatorBase:
        pass

    @abstractmethod
    def _update_nonadaptive_ml_estimator(estimator, new_max_level, mse_tol, init_nsamp) -> MLMCNonAdaptiveEstimatorBase:
        pass

    def _save_results(self, final_ml_estimator, final_max_level):
        self._final_estimator = final_ml_estimator
        self._estimate = final_ml_estimator.estimate
        self._nsamp_per_level = final_ml_estimator.nsamp_per_level
        self._max_level_used = final_max_level

    def _conv_check(self, max_level, ml_estimator, mse_tol):
        if (self._is_bias_under_tol(ml_estimator.est_per_level_adjusted[-1], mse_tol) and self._is_variance_under_tol(ml_estimator, mse_tol)):
            conv_success = True
        else:
            conv_success = False
        if (not self._is_bias_under_tol(ml_estimator.est_per_level_adjusted[-1], mse_tol)):
            max_level += 1
        return max_level, conv_success

    def _is_bias_under_tol(self, finest_level_est, mse_tol): #TODO If nsamp on finest level is too low, innacurate. Extrapolate from the previous level in that case.
        m = self._model.m #TODO
        if abs(finest_level_est) <= ((self._r*(m**self._alpha)-1)/np.sqrt(2))*np.sqrt(mse_tol):
            return True
        else:
            return False

    def _is_variance_under_tol(self, ml_estimator, mse_tol):
        num_samp = ml_estimator.nsamp_per_level
        var_per_level = ml_estimator.var_per_level_adjusted
        sample_var = np.sum(var_per_level / num_samp)
        if sample_var <= mse_tol/2:
            return True
        else:
            return False
