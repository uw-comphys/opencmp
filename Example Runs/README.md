## README

Each folder contains the config file and model functions required to run a particular example. Odd-numbered examples use standard continuous finite elements, while even-numbered examples use DG.



**Poisson Equation:**

* poisson_1/2: Tests h-convergence.
* poisson_3/4/5/6/7/8: Tests a transient solve using explicit Euler, Crank-Nicolson, and implicit Euler respectively.
* poisson_9/10: Tests adaptive time-stepping with the two-step scheme.
* poisson_11/12: Tests adaptive time-stepping with the three-step scheme.

**Stokes Flow:**

* stokes_1/2: Steady-state channel flow. Shows that pressure BCs can work when the exact solution is known. Calculates L1, L2, L$\infty$ norm, divergence of u, and facet jumps in u and p.

**Incompressible Navier-Stokes:**

* ins_1/2: Tests a transient solve (vortex-shedding). No exact solution, but shedding should start after ~0.5s.
* ins_3/4: Steady-state channel flow. Shows that pressure BCs can work when the exact solution is known.
* ins_5/6: Steady-state channel flow. Calculates L1, L2, L$\infty$ norm, divergence of u, and facet jumps in u and p.
* ins_7/8: Transient INS with a known solution.
* ins_9/10, 11/12, 13/14: Tests a transient solve (vortex-shedding) using various IMEX methods (CNLF, SBDF, and adaptive IMEX respectively). Should give the same results as ins_1/2.

**Diffuse Interface Method:** 

*DIM is not fully implemented in any model, these examples are only meant to test generating the phase fields*

* DIM_poisson_1: Poisson solve with the phase field, mesh, and masks generated from a .stl file.
* DIM_poisson_2: Poisson solve with the phase field, mesh, and masks loaded from .sol/.vol files.
* DIM_poisson_3: Poisson solve that inverts the phase field.
* DIM_poisson_4: Poisson solve that combines multiple .stl files into one phase field.
* DIM_poisson_5: Copy of DIM_poisson_1 but uses a coarse mesh for the simulation and a refined mesh for constructing the phase field.
* DIM_poisson_6: Copy of DIM_poisson_1 that uses DG.
* DIM_stokes_1: DG Stokes solve with the phase field, mesh, and masks generated from a .stl file.
* DIM_stokes_2: DG Stokes solve with the phase field, mesh, and masks loaded from .sol/.vol files.



When creating config files:

* -> is the standard delimiter for a two-level parameter.
  * ex: u = top -> 2*x
* Coordinates are indicated by <>.
  * ex: The origin would be <0.0, 0.0> in 2D.
* Vector expressions are indicated by [].
  * ex: A velocity coefficient function might be [sin(pi\*x), cos(pi\*y), z] in 3D.



The following libraries are required to use OpenCMP. 

* NGSolve version 6.2.2101 or later.
* edt (`pip install edt`)
* tabulate (`pip install tabulate`)
* meshio (`pip install meshio`)