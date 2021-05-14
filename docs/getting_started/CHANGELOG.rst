OpenCMP Changelog
=================

May 14

* Added the multicomponent INS model.
* Boundary condition values, model parameter values, etc are now saved for every step in the time discretization scheme. This fixes an implementation error that was causing suboptimal time step convergence rates for the higher-order time discretization schemes.
* The three-step adaptive time-stepping scheme is currently broken.
* Added Runge Kutta IMEX schemes, but they don't entirely work yet.

April 30

* Added option to use no preconditioner (specify preconditioner as "None").
* Several improvements to the code for the diffuse interface method.
    
    - To improve the fidelity of the phase field values outside of the expected range of [0,1] are now truncated to that range.
    - Added option to load :math:`\phi`, :math:`\underline{\nabla} \phi`, and :math:`\lvert \underline{\nabla} \phi \rvert` all from file instead of loading only :math:`\phi` from file and then generating the remaining phase fields from it.
    - Added option to generate structured triangular/tetrahedral meshes so adaptive mesh refinement can be used.

* Removed "pinned" boundary condition from the Poisson model. Use a Dirichlet boundary condition instead.
* Minor bug fixes and changes to/additional unit tests.

April 23

* Added multi-component INS model and examples.
* Changed the formatting of the pytest integration tests.

