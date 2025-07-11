from core.model.model_base import ModelBase, ModelEvaluationBase
from core.sample.sample_base import SampleBase
import numpy as np
from dolfinx.fem.petsc import LinearProblem
from setup.lognormal_pde_setup import PDEwLognormalRandomCoeff as PDE


class LognormalPDESample(SampleBase):

    def __init__(self, rng: np.random.Generator):
        self._s = 4
        self._std = 2
        self._rng = rng

    def draw(self):
        return self._std*self._rng.standard_normal(self._s)
    

class LognormalPDEEvaluation(ModelEvaluationBase):

    def __init__(self, value, cost):
        self._value = value
        self._cost = cost

    @property
    def value(self):
        return self._value
    
    @property
    def cost(self):
        return self._cost


class LognormalPDEModel(ModelBase):
    
    def __init__(self, visualise=False):
        self._h0 = 0.25
        self._m = 2.
        self._decay_rate_q = 2.
        self._visualise = visualise

    def evaluate(self, level, sample) -> ModelEvaluationBase:
        hl = self.get_hl(level)
        a, L, bc = PDE.setup_fenics_problem(hl, sample, self._decay_rate_q)
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=self._get_petsc_options(), jit_options={"timeout": 60})
        uh = problem.solve()
        qoi = self._get_qoi_from_solution(uh, level)
        evaluation_cost = self._get_eval_cost(level)
        return LognormalPDEEvaluation(qoi, evaluation_cost)
    
    def get_hl(self, level):
        return self._h0*(self._m**(-level))
    
    def _get_petsc_options(self):
        return {"ksp_type": "preonly", "pc_type": "lu"}
    
    def _get_qoi_from_solution(self, uh, level):
        uh_vec = uh.x.array
        hl = self.get_hl(level)
        return (hl**2)*sum(uh_vec)

    def _get_eval_cost(self, level):
        matrix_order = ((1/self.get_hl(level))-1)**2
        return matrix_order
    
    @property
    def m(self):
        return self._m
    
    @property
    def decay_rate_q(self):
        return self._decay_rate_q
