import numpy as np
from dolfinx import fem, mesh, plot
from mpi4py import MPI
import ufl
from ufl import dx, grad, inner
from petsc4py.PETSc import ScalarType
import pyvista as pv


class PDEwLognormalRandomCoeff():

    @staticmethod
    def get_mesh(hl):
        num_el = int(np.round(1/hl))
        msh = mesh.create_rectangle(
                comm=MPI.COMM_SELF,
                points=((0.0, 0.0), (1.0, 1.0)),
                n=(num_el, num_el),
                cell_type=mesh.CellType.triangle,
                )
        return msh
    
    @staticmethod
    def get_pde_coeff(x, sample, decay_rate_q):
        coeff = ufl.exp(
            sum(
                sample[j] * (1 / (j + 1) ** decay_rate_q) * ufl.sin(2 * np.pi * (j + 1) * x[0]) * ufl.cos(2 * np.pi * (j + 1) * x[1])
                for j in range(len(sample))
            )
        )
        return coeff
    
    @staticmethod
    def get_V(msh):
        return fem.functionspace(msh, ("Lagrange", 1))
    
    @staticmethod
    def get_x(msh):
        return ufl.SpatialCoordinate(msh)
    
    @staticmethod
    def get_source_term(msh):
        return fem.Constant(msh, ScalarType(1.0))
    
    @staticmethod
    def get_trial_fction_u(V):
        return ufl.TrialFunction(V)
    
    @staticmethod
    def get_test_fction_v(V):
        return ufl.TestFunction(V)
    
    @staticmethod
    def get_bilin_form_a(coeff, u, v):
        return inner(coeff * grad(u), grad(v)) * dx
    
    @staticmethod
    def get_lin_form_l(f, v):
        return inner(f, v) * dx
    
    @staticmethod
    def get_boundary_facets(msh):
        return mesh.locate_entities_boundary(
                msh,
                dim=(msh.topology.dim - 1),
                marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
                )
    
    @staticmethod
    def get_boundary_dofs(V, boundary_facets):
        return fem.locate_dofs_topological(V, entity_dim=1, entities=boundary_facets)
    
    @staticmethod
    def get_bc(boundary_dofs, V):
        return fem.dirichletbc(value=ScalarType(0), dofs=boundary_dofs, V=V)
    
    @staticmethod
    def setup_fenics_problem(hl, sample, decay_rate_q):
        msh = PDEwLognormalRandomCoeff.get_mesh(hl)
        V = PDEwLognormalRandomCoeff.get_V(msh)
        x = PDEwLognormalRandomCoeff.get_x(msh)
        coeff = PDEwLognormalRandomCoeff.get_pde_coeff(x, sample, decay_rate_q)
        f = PDEwLognormalRandomCoeff.get_source_term(msh)
        u = PDEwLognormalRandomCoeff.get_trial_fction_u(V)
        v = PDEwLognormalRandomCoeff.get_test_fction_v(V)
        a = PDEwLognormalRandomCoeff.get_bilin_form_a(coeff, u, v)
        L = PDEwLognormalRandomCoeff.get_lin_form_l(f, v)
        boundary_facets = PDEwLognormalRandomCoeff.get_boundary_facets(msh)
        boundary_dofs = PDEwLognormalRandomCoeff.get_boundary_dofs(V, boundary_facets)
        bc = PDEwLognormalRandomCoeff.get_bc(boundary_dofs, V)
        return a, L, bc
    
    def plot_solution(self, hl, sample):
        raise NotImplementedError
        # Create a VTK-compatible representation of the mesh
        V = self.get_V()
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