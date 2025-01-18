from estimator.mlmc_estimator import MLMCNonAdaptiveEstimator, MLMCAdaptiveEstimator
import numpy as np
from model.model_base import MPMLModel


class MPMLNonAdaptiveEstimator(MLMCNonAdaptiveEstimator):

    def __init__(self, sample, model: MPMLModel, nsamp_per_level, Lmin, comp_tol_per_level):
        super().__init__(sample, model, nsamp_per_level, Lmin)
        self._comp_tol_per_level = comp_tol_per_level

    def _evaluate_model(self, level, sample):
        return self._model.evaluate(level, sample, comp_tol=self.comp_tol_per_level[level-self._Lmin])

    @property
    def comp_tol_per_level(self):
        return self._comp_tol_per_level
    

class MPMLAdaptiveEstimator(MLMCAdaptiveEstimator):

    def __init__(self, sample, model: MPMLModel, Lmin, Lmax, alpha, beta, alpha_tol, beta_tol, k_p=0.1):
        super().__init__(sample, model, Lmin, Lmax, alpha, beta)
        self._k_p = k_p
        self._alpha_tol = alpha_tol
        self._beta_tol = beta_tol

    def _setup_nonadaptive_ml_estimator(self, nsamp, max_level) -> MPMLNonAdaptiveEstimator:
        return MPMLNonAdaptiveEstimator(self._sample, self._model, nsamp*np.ones(max_level-self._Lmin+1), self._Lmin, self._get_comp_tol(max_level))
    
    def _update_nonadaptive_ml_estimator(self, estimator, new_max_level, mse_tol, init_nsamp):
        aux = super()._update_nonadaptive_ml_estimator(estimator, new_max_level, mse_tol, init_nsamp)
        comp_tol = self._get_comp_tol(new_max_level)
        return MPMLNonAdaptiveEstimator(self._sample, self._model, aux.nsamp_per_level, self._Lmin, comp_tol)
    
    def _get_comp_tol(self, cur_max_level):
        comp_tol = np.zeros(cur_max_level-self._Lmin+1)
        for i in range(self._Lmin, cur_max_level):
            comp_tol[i-self._Lmin] = ((self._k_p/2)*(self._model.get_hl(i)**self._beta))**(1/self._beta_tol)
        comp_tol[cur_max_level-self._Lmin] = min(((self._k_p/2)*(self._model.get_hl(cur_max_level)**self._beta))**(1/self._beta_tol),
                                                 ((self._k_p/2)*(self._model.get_hl(cur_max_level)**self._alpha))**(1/self._alpha_tol))
        return comp_tol
