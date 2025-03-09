from core.estimator.estimator_base import MLMCAdaptiveEstimatorBase, MLMCNonAdaptiveEstimatorBase
from core.model.model_base import ModelBase
from core.sample.sample_base import SampleBase
import numpy as np
from copy import deepcopy
import logging


class MLMCNonAdaptiveEstimator(MLMCNonAdaptiveEstimatorBase):

    def __init__(self, sample: SampleBase, model: ModelBase, nsamp_per_level: np.array, Lmin, mc_diff_per_level=None, precomputed_samples_cost = None):
        self._sample = sample
        self._model = model
        self._nsamp_per_level = nsamp_per_level
        self._Lmin = Lmin
        self._Lmax = Lmin + len(nsamp_per_level) - 1
        self._reset_results()
        if mc_diff_per_level is not None:
            if type(mc_diff_per_level) is not list:
                raise ValueError("Given precomputed values for the MLMC estimator have the wrong type. The correct type is list of np arrays.")
            if len(mc_diff_per_level) != len(nsamp_per_level):
                raise ValueError("Given list of precomputed samples must have the same length as number of levels.")
            for i in range(len(nsamp_per_level)):
                if len(mc_diff_per_level[i]) > nsamp_per_level[i]:
                    raise ValueError(f"Maximum number of precomputed values cannot be bigger than number of samples on level {i+Lmin}")
            if len(mc_diff_per_level) != len(precomputed_samples_cost):
                raise ValueError("Lenght of precomputed samples list must be the same as length of the costs.")
            self._mc_differences_per_level = mc_diff_per_level
            self._precomputed_samples_cost = precomputed_samples_cost
        else:
            self._mc_differences_per_level = []
            self._precomputed_samples_cost = []
            for i in range(len(nsamp_per_level)):
                self._mc_differences_per_level.append(np.array([]))
                self._precomputed_samples_cost.append(0)

    def _reset_results(self):
        self._run_success = None
        self._estimate = None
        self._est_per_level = None
        self._var_per_level = None
        self._est_per_level_adjusted = None
        self._var_per_level_adjusted = None
        self._cost_per_level_per_sample = None

    def _run_impl(self, save_mc_differences=False, **kwargs):
        self._reset_results()

        # Initialize containers to store results
        self._est_per_level = []
        self._var_per_level = []
        self._cost_per_level_per_sample = []

        # Iterate through levels from Lmin to Lmax
        for level, nsamp in enumerate(self._nsamp_per_level.astype(int), start=self._Lmin):
            nsamp_to_compute = nsamp - len(self._mc_differences_per_level[level-self._Lmin])
            # Compute model outputs for the current level
            mc_differences, cost_per_sample = self._get_mc_differences(level, nsamp_to_compute, **kwargs)
            self._cost_per_level_per_sample.append(np.average([self._precomputed_samples_cost[level - self._Lmin], cost_per_sample],
                                                              weights=[1-(nsamp_to_compute/nsamp), nsamp_to_compute/nsamp]))
            self._mc_differences_per_level[level-self._Lmin] = np.append(self._mc_differences_per_level[level-self._Lmin], mc_differences)

            # Compute estimates and variances for the current level
            level_estimate = np.mean(self._mc_differences_per_level[level-self._Lmin])
            level_variance = np.var(self._mc_differences_per_level[level-self._Lmin], ddof=0)

            # Store the results
            self._est_per_level.append(level_estimate)
            self._var_per_level.append(level_variance)

        # Compute the overall MLMC estimate
        self._estimate = sum(self._est_per_level)
        self._run_success = True

    def _get_mc_differences(self, level, nsamp, **kwargs):
        mc_differences = []
        level_cost = []
        for i in range(nsamp):
            sample = self._sample.draw()
            if level == self._Lmin:
                evaluation = self._evaluate_model(level, sample, **kwargs)
                mc_differences.append(evaluation.value)
                level_cost.append(evaluation.cost)
            else:
                eval_fine = self._evaluate_model(level, sample, **kwargs)
                eval_coarse = self._evaluate_model(level-1, sample, **kwargs)
                mc_differences.append(eval_fine.value - eval_coarse.value)
                level_cost.append(eval_fine.cost + eval_coarse.cost)
        cost_per_sample = np.mean(level_cost) if nsamp != 0 else 0
        return np.array(mc_differences), np.array(cost_per_sample)

    def _evaluate_model(self, level, sample):
        return self._model.evaluate(level, sample)
    
    def adjust_estimates_and_variances(self, alpha, beta):
        self._est_per_level_adjusted = deepcopy(self._est_per_level)
        self._var_per_level_adjusted = deepcopy(self._var_per_level)
        for level in range(self._Lmin+1, self._Lmax+1):
            if self._var_per_level_adjusted[level-self._Lmin] > self._var_per_level_adjusted[level-self._Lmin-1]:
                self._var_per_level_adjusted[level-self._Lmin] = self._var_per_level_adjusted[level-self._Lmin-1]/(0.5*self._model.m**beta)
        for level in range(self._Lmin+1, self._Lmax+1):
            self._est_per_level_adjusted[level-self._Lmin] = max(self._est_per_level_adjusted[level-self._Lmin], 
                                                                 self._est_per_level_adjusted[level-self._Lmin-1]/(self._model.m**alpha))
            self._var_per_level_adjusted[level-self._Lmin] = max(self._var_per_level_adjusted[level-self._Lmin], 
                                                                 self._var_per_level_adjusted[level-self._Lmin-1]/(0.5*self._model.m**beta))

    @property
    def nsamp_per_level(self):
        return self._nsamp_per_level
    
    @nsamp_per_level.setter
    def nsamp_per_level(self, value):
        if not isinstance(value, (list, tuple, np.array)):
            raise ValueError("nsamp_per_level must be a list, tuple, or numpy array.")
        if min(value) <= 0:
            raise ValueError("number of samples must be positive")
        self._nsamp_per_level = value

    @property
    def est_per_level(self):
        return self._est_per_level

    @property
    def var_per_level(self):
        return self._var_per_level
    
    @property
    def est_per_level_adjusted(self):
        if self._est_per_level_adjusted == None:
            raise RuntimeError("Estimates have not been adjusted.")
        return self._est_per_level_adjusted
    
    @property
    def var_per_level_adjusted(self):
        if self._var_per_level_adjusted == None:
            raise RuntimeError("Variances have not been adjusted.")
        return self._var_per_level_adjusted
    
    @property
    def cost_per_level_per_sample(self):
        return self._cost_per_level_per_sample
    
    @property
    def estimate(self):
        return self._estimate
    
    @property
    def nsamp_per_level(self):
        return self._nsamp_per_level
    
    @property
    def mc_differences_per_level(self):
        return self._mc_differences_per_level


class MLMCAdaptiveEstimator(MLMCAdaptiveEstimatorBase):

    def _setup_nonadaptive_ml_estimator(self, nsamp, max_level) -> MLMCNonAdaptiveEstimator:
        return MLMCNonAdaptiveEstimator(self._sample, self._model, nsamp*np.ones(max_level-self._Lmin+1), self._Lmin)
    
    def _update_nonadaptive_ml_estimator(self, estimator: MLMCNonAdaptiveEstimator, new_max_level, mse_tol) -> MLMCNonAdaptiveEstimator:
        nsamp_old = deepcopy(estimator.nsamp_per_level)
        precomputed_samples = []
        precomputed_samples_cost = estimator.cost_per_level_per_sample
        nsamp_new = []
        const_ = (2/mse_tol)*np.sum(np.sqrt(np.array(estimator.var_per_level_adjusted)*np.array(estimator.cost_per_level_per_sample)))
        for i in range(len(nsamp_old)):
            nsamp_new.append(max(np.ceil(const_*np.sqrt(estimator.var_per_level_adjusted[i]/estimator.cost_per_level_per_sample[i])), self._min_nsamp))
            old_samples = estimator.mc_differences_per_level[i]
            precomputed_samples.append(old_samples[:min(int(nsamp_new[i]), len(old_samples))])
        if new_max_level > estimator._Lmax:
            var_extrapolated = estimator.var_per_level_adjusted[-1]/(self._model.m**self._beta)
            cost_extrapolated = estimator.cost_per_level_per_sample[-1]*(self._model.m**self._cost_extrapol_const)
            nsamp_new.append(max(np.ceil(const_*np.sqrt(var_extrapolated/cost_extrapolated)), self._min_nsamp))
            precomputed_samples.append(np.array([]))
            precomputed_samples_cost.append(0)
        for i in range(1, len(nsamp_new)):
            if nsamp_new[i] > nsamp_new[i - 1]:
                logging.warning(f"Possibly erratic number of samples: {nsamp_new}")
        return MLMCNonAdaptiveEstimator(self._sample, self._model, np.array(nsamp_new).astype(int), self._Lmin, mc_diff_per_level=precomputed_samples, precomputed_samples_cost=precomputed_samples_cost)