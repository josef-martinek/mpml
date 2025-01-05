from model.model_base import ModelBase
from sample.sample_base import SampleBase
import numpy as np
from abc import ABC, abstractmethod


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

    @abstractmethod
    def cost_per_level_per_sample(self):
        pass

    @abstractmethod
    def run(self):
        pass


class MLMCAdaptiveEstimatorBase(ABC):
    def __init__(self, sample: SampleBase, model: ModelBase, Lmin, Lmax, alpha, beta, gamma, **kwargs):
        self._sample = sample
        self._model = model
        if Lmin >= Lmax:
            raise ValueError("Estimator must have at least 2 levels.")
        self._Lmin = Lmin
        self._Lmax = Lmax
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._r = 0.7
        self._reset_results()

    def _reset_results(self):
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

    def run(self, mse_tol, init_nsamp=10):
        self._reset_results()
        cur_max_level = self._Lmin + 1
        cur_ml_estimator = self._setup_nonadaptive_ml_estimator(init_nsamp, cur_max_level)
        while cur_max_level <= self._Lmax:
            #est_per_level, var_per_level, cost_per_level = cur_ml_estimator.get_ml_estimates_per_level()
            cur_ml_estimator.run()
            new_max_level, conv_success = self._conv_check(cur_max_level, cur_ml_estimator, mse_tol)
            if conv_success:
                print("convergence successful")
                self._conv_success = True
                self._save_results(cur_ml_estimator, cur_max_level)
            if new_max_level <= self._Lmax: 
                cur_ml_estimator = self._update_nonadaptive_ml_estimator(cur_ml_estimator, new_max_level, mse_tol, init_nsamp)
            cur_max_level = new_max_level
        self._conv_success = False
        if cur_max_level > self._Lmax:
            print("Convergence not achieved, max level exceeded")
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
        self._estimate = final_ml_estimator.estimate()
        self._nsamp_per_level = final_ml_estimator.nsamp_per_level()
        self._max_level_used = final_max_level

    def _conv_check(self, max_level, ml_estimator, mse_tol): 
        #If bias is not under tol, increase max level. If both bias and variance are under tol, finish.
        if (not self._is_bias_under_tol(ml_estimator.est_per_level[-1], mse_tol)):
            max_level += 1
            conv_success = False
        if (self._is_bias_under_tol(ml_estimator.est_per_level[-1], mse_tol) and self._is_variance_under_tol(ml_estimator, mse_tol)):
            conv_success = True
        return max_level, conv_success

    def _is_bias_under_tol(self, finest_level_est, mse_tol):
        m = self._model.m() #TODO
        if abs(finest_level_est) <= ((self._r*(m**self._alpha)-1)/np.sqrt(2))*np.sqrt(mse_tol):
            return True
        else:
            return False

    def _is_variance_under_tol(self, ml_estimator, mse_tol):
        num_samp = ml_estimator.nsamp_per_level()
        var_per_level = ml_estimator.var_per_level()
        sample_var = np.sum(var_per_level / num_samp)
        if sample_var <= mse_tol/2:
            return True
        else:
            return False
