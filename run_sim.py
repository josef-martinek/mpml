from model.model_base import MLMCModel
import numpy as np

class MLMCNonAdaptiveEstimator:
     def __init__(self, parameter, model: MLMCModel, Lmin, nsamp: np.array):
          raise NotImplementedError
     
     def get_ml_estimates_per_level(self):
          raise NotImplementedError
     


class MLMCAdaptiveEstimator:
    def __init__(self, parameter, model: MLMCModel, Lmin , Lmax):
        self._parameter = parameter
        self._model = model
        if Lmin >= Lmax:
            raise ValueError("Estimator must have at least 2 levels.")
        self._Lmin = Lmin
        self._Lmax = Lmax
        self._reset_results()

    def _reset_results(self):
            self._conv_success = None
            self._estimate = None
            self._n_samp = None
            self._n_levels_used = None

    def run(self, mse_tol, init_nsamp = 10):
        self._reset_results()
        cur_max_level = self._Lmin + 1
        cur_nsamp = init_nsamp*np.ones(1,2)
        while cur_max_level <= self._Lmax:
            cur_ml_estimator = MLMCNonAdaptiveEstimator(self._parameter, self._model, self._Lmin, cur_nsamp)
            est_per_level, var_per_level, cost_per_level = cur_ml_estimator.get_ml_estimates_per_level(cur_nsamp)
            new_max_level, conv_success = self._conv_check(est_per_level, var_per_level, cost_per_level, mse_tol)
            if conv_success: #TODO return results
                 print("convergence successful")
                 self._conv_success = True
            cur_nsamp = self._update_nsamp()
            if new_max_level > cur_max_level:
                 np.append(cur_nsamp, init_nsamp)
            cur_max_level = new_max_level
        self._conv_success = False
        if cur_max_level > self._Lmax:
             raise RuntimeError("Convergence not achieved, max level exceeded")
            #TODO return result anyway
        else:
             raise RuntimeError("Convergence not achieved, but max level not exceeded. That means bug.")
    
    def _conv_check(self, est_per_level_var_per_level, cost_per_level, mse_tol): 
        #If bias is not under tol, increase max level. If this increases the level above Lmax, end. -> Do this probably in the while loop. If both bias and variance are under tol, finish.
        raise NotImplementedError
        if (not self._is_bias_under_tol()):
                cur_max_level += 1
        if (self._is_bias_under_tol() and self._is_variance_under_tol()) #TODO
    
    def _is_bias_under_tol(self, finest_level_est, bias_tol):
         raise NotImplementedError
    
    def _is_variance_under_tol(self, mlmc_sample_var, var_tol):
         raise NotImplementedError
    
    def _update_nsamp(self):
         raise NotImplementedError