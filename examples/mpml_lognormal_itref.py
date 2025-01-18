from model.model_base import MPMLModel, ModelEvaluationBase
from examples.mlmc_lognormal import LognormalPDEModel, LognormalPDEEvaluation
from dolfinx.fem import form, apply_lifting, Function, functionspace
from dolfinx.fem import assemble_matrix, assemble_vector
import logging
from examples.lognormal_pde_setup import PDEwLognormalRandomCoeff as PDE
from linsolver.itref import itref
from dolfinx import la
import scipy as sp
import numpy as np


class MPLognormalPDEModelItref(MPMLModel, LognormalPDEModel):

    def evaluate(self, level, sample, comp_tol) -> ModelEvaluationBase:
        hl = self.get_hl(level)
        a, L, bc = PDE.setup_fenics_problem(hl, sample, self._decay_rate_q)
        a = form(a)
        L = form(L)
        A = assemble_matrix(a, bcs=[bc])
        #A.assemble()
        b = assemble_vector(L)
        #b.assemble()
        apply_lifting(b.array, [a], bcs=[[bc]])
        bc.set(b.array)

        itref_l = self._get_itref(hl)
        x, num_it, rel_res = itref_l.solve_system(A=self._csr_mat_to_ndarray(A), b=self._vec_to_ndarray(b), rtol=comp_tol)

        logging.trace(f'Itref iteration number on level {level} with hl {hl} is {num_it}')
        logging.trace(f'Computational tolerance: {comp_tol}')
        logging.trace(f'Relative residual norms: {rel_res}\n')
        if rel_res[-1] > comp_tol:
            logging.warning(f'Itref not converged within given max number of iterations. Rel. residual: {rel_res[-1]}. Tolerance: {comp_tol}.\n')
       
        msh = PDE.get_mesh(hl)
        uh = Function(functionspace(msh, ("Lagrange", 1)))
        uh.x.array[:] = x[:]
        qoi = self._get_qoi_from_solution(uh, level)
        evaluation_cost = self._get_eval_cost(hl)
        return LognormalPDEEvaluation(qoi, evaluation_cost)
    
    def _get_eval_cost(self, hl):
        matrix_order = ((1/hl)-1)**2
        return matrix_order**(3/2)
    
    @staticmethod
    def _csr_mat_to_ndarray(A) -> np.ndarray:
        return A.to_dense()
    
    @staticmethod
    def _vec_to_ndarray(v) -> np.ndarray:
        return v.array
    
    @staticmethod
    def _get_itref(hl): #TODO switch to hl
        if np.allclose(hl, 1/4):
            return itref(np.half, np.half, np.half, np.half)
        elif np.allclose(hl, 1/8):
            return itref(np.half, np.half, np.single, np.single)
        elif np.allclose(hl, 1/16):
            return itref(np.half, np.half, np.single, np.single)
        else:
            return itref(np.single, np.single, np.double, np.double)