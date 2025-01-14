from model.model_base import ModelBase, ModelEvaluationBase
from sample.sample_base import SampleBase
import dolfinx
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import dx, grad, inner
import pyvista as pv


class LognormalPDESample(SampleBase):

    def __init__(self):
        self._s = 4
        self._std = np.sqrt(1.4)

    def draw(self):
        return self._std*np.random.normal(0, 1, self._s)
    

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
        self._h0 = 1/4
        self._m = 2
        self._q = 2
        self._visualise = visualise

    def evaluate(self, level, sample) -> ModelEvaluationBase:
        a, L, bc = self._setup_fenicsx_problem(level, sample)
        problem = LinearProblem(a, L, bcs=[bc], petsc_options=self._get_petsc_options())
        uh = problem.solve()
        qoi = self._get_qoi_from_solution(uh, level)
        if self._visualise:
            self._plot_solution(uh, level)
        evaluation_cost = self._get_eval_cost(level)
        return LognormalPDEEvaluation(qoi, evaluation_cost)

    def _get_mesh(self, level):
        num_el = int(np.round(1/self.get_hl(level)))
        msh = mesh.create_rectangle(
                comm=MPI.COMM_WORLD,
                points=((0.0, 0.0), (1.0, 1.0)),
                n=(num_el, num_el),
                cell_type=mesh.CellType.triangle,
                )
        return msh
    
    def get_hl(self, level):
        return self._h0*(self._m**(-level))

    def _get_pde_coeff(self, x, sample):
        coeff = ufl.exp(
            sum(
                sample[j] * (1 / (j + 1) ** self._q) * ufl.sin(2 * np.pi * (j + 1) * x[0]) * ufl.cos(2 * np.pi * (j + 1) * x[1])
                for j in range(len(sample))
            )
        )
        return coeff
    
    def _setup_fenicsx_problem(self, level, sample):
        msh = self._get_mesh(level)
        V = fem.functionspace(msh, ("Lagrange", 1))
        x = ufl.SpatialCoordinate(msh)
        coeff_a = self._get_pde_coeff(x, sample)

        # Define source term f
        f = fem.Constant(msh, ScalarType(1.0))

        # Define trial and test functions
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)

        # Define the bilinear form and linear form
        a = inner(coeff_a * grad(u), grad(v)) * dx
        L = inner(f, v) * dx

        boundary_facets = mesh.locate_entities_boundary(
            msh,
            dim=(msh.topology.dim - 1),
            marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
        )
        boundary_dofs = fem.locate_dofs_topological(V, entity_dim=1, entities=boundary_facets)
        bc = fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)
        return a, L, bc
    
    def _get_petsc_options(self):
        return {"ksp_type": "preonly", "pc_type": "lu"}
    
    def _get_qoi_from_solution(self, uh, level):
        uh_vec = uh.x.array
        hl = self.get_hl(level)
        return (hl**2)*sum(uh_vec)
    
    def _plot_solution(self, uh, level):
        # Create a VTK-compatible representation of the mesh
        V = fem.functionspace(self._get_mesh(level), ("Lagrange", 1))
        cells, types, x = plot.vtk_mesh(V)

        # Create a PyVista grid and attach the solution values
        grid = pv.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = uh.x.array.real
        grid.set_active_scalars("u")
        # Plot the solution
        plotter = pv.Plotter()
        #warped = grid.warp_by_scalar()
        #plotter.add_mesh(warped)
        plotter.add_mesh(grid, scalars="u", cmap="viridis", show_edges=True)
        #plotter.add_scalar_bar(title="u(x, y)", n_labels=5)
        plotter.view_xy()
        plotter.show()

    def _get_eval_cost(self, level):
        matrix_order = ((1/self.get_hl(level))-1)**2
        return matrix_order**(3/2)
    
    @property
    def m(self):
        return self._m