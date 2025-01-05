from estimator.estimator_base import MLMCAdaptiveEstimatorBase, MLMCNonAdaptiveEstimatorBase
from model.model_base import ModelBase
from sample.sample_base import SampleBase
import numpy as np
from copy import deepcopy

class MLMCNonAdaptiveEstimator(MLMCNonAdaptiveEstimatorBase):

    def __init__(self, sample: SampleBase, model: ModelBase, nsamp_per_level: np.array, Lmin):
        self._sample = sample
        self._model = model
        self._nsamp_per_level = nsamp_per_level
        self._Lmin = Lmin
        self._Lmax = Lmin + len(nsamp_per_level) - 1
        self._reset_results()

    def _reset_results(self):
        self._run_success = None
        self._estimate = None
        self._est_per_level = None
        self._var_per_level = None
        self._est_per_level_adjusted = None
        self._var_per_level_adjusted = None
        self._cost_per_level_per_sample = None
        self._samples_per_level = None
        self._mc_differences_per_level = None

    def run(self, save_samples=False, save_mc_differences=False):
        self._reset_results()

        # Initialize containers to store results
        self._est_per_level = []
        self._var_per_level = []
        self._cost_per_level_per_sample = []
        self._samples_per_level = [] if save_samples else None
        self._mc_differences_per_level = [] if save_mc_differences else None

        # Iterate through levels from Lmin to Lmax
        for level, nsamp in enumerate(self._nsamp_per_level, start=self._Lmin):
            # Generate samples for the current level
            samples = self._sample.generate(nsamp)
            if save_samples:
                self._samples_per_level.append(samples)

            # Compute model outputs for the current level
            mc_differences, cost_per_sample = self._get_mc_differences(level, samples)
            self._cost_per_level_per_sample.append(cost_per_sample)
            if save_mc_differences:
                self._mc_differences_per_level.append(mc_differences)

            # Compute estimates and variances for the current level
            level_estimate = np.mean(mc_differences)
            level_variance = np.var(mc_differences, ddof=1)  # Unbiased variance

            # Store the results
            self._est_per_level.append(level_estimate)
            self._var_per_level.append(level_variance)

        # Compute the overall MLMC estimate
        self._estimate = sum(self._est_per_level)
        self._run_success = True

    def _get_mc_differences(self, level, samples):
        mc_differences = []
        level_cost = []
        for sample in samples:
            if level == self._Lmin:
                evaluation = self._model.evaluate(level, sample)
                mc_differences.append(evaluation.value)
                level_cost.append(evaluation.cost)
            else:
                eval_fine = self._model.evaluate(level, sample)
                eval_coarse = self._model.evaluate(level-1, sample)
                mc_differences.append(eval_fine.value - eval_coarse.value)
                level_cost.append(eval_fine.cost + eval_coarse.cost)
        cost_per_sample = np.mean(level_cost)
        return mc_differences, cost_per_sample
    
    def adjust_estimates_and_variances(self, alpha, beta):
        self._est_per_level_adjusted = deepcopy(self._est_per_level)
        self._var_per_level_adjusted = deepcopy(self._var_per_level)
        if len(self._nsamp_per_level) < 3:
            pass
        else:
            for level in range(2, self._Lmax+1):
                self._est_per_level_adjusted[level] = max(self._est_per_level_adjusted[level], self._est_per_level_adjusted[level-1]/(self._model.m**alpha))
                self._var_per_level_adjusted[level] = max(self._var_per_level_adjusted[level], self._var_per_level_adjusted[level-1]/(self._model.m**beta))

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


class MLMCAdaptiveEstimator(MLMCAdaptiveEstimatorBase):

    def _setup_nonadaptive_ml_estimator(self, nsamp, max_level):
        return MLMCNonAdaptiveEstimator(self._sample, self._model, nsamp*np.ones(1,max_level-self._Lmin+1), self._Lmin)
    
    def _update_nonadaptive_ml_estimator(self, estimator: MLMCNonAdaptiveEstimator, new_max_level, mse_tol, init_nsamp):
        #First, extrapolate var per level if it's zero at some point, based on tha, then update nsamp. If new max level is higher than old max level, add new level with init_nsamp.
        nsamp_old = deepcopy(estimator.nsamp_per_level)
        nsamp_new = []
        const_ = (2/mse_tol)*np.sum(np.sqrt(estimator.var_per_level_adjusted*estimator.cost_per_level_per_sample))
        for i in range(len(nsamp_old)):
            nsamp_new.append(np.ceil(const_*np.sqrt(estimator.var_per_level_adjusted[i]/estimator.cost_per_level_per_sample[i])))
        if new_max_level > estimator._Lmax:
            nsamp_new.append(init_nsamp)
        return MLMCNonAdaptiveEstimator(self._sample, self._model, nsamp_new, self._Lmin)