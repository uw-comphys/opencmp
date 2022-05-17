OpenCMP Changelog
=================

May 17, 2022

* Added detection of underlying NGSolve linear solver configuration and changed default direct linear solver to PARDISO.
* Slight modification to diffuse interface code to account for namespace changes in SciPy.

February 8, 2022

* Reordered examples so that the previous Example 1 is now part of the getting-started documented, shifted all remaining examples down by 1.
* Corrected typo in Example 6, now Example 5.
* Added Example 10, based on an existing 3D immersed cylinder flow benchmark.
* Corrected issues with --runslow option for pytests
* Minor refactoring of code.
* Fixed an issue with the config parser that caused it to crash when a single boundary was given to the surface traction calculation code.
* Changed `sol_to_vtu()` so that by default it will respect the `num_threads` setting in the config file. Added an optional parameter that can be used to override this behaviour.
* Replaced some of the bare `except` statements with either the specific error they catch, or reworked the code to not use `try` and `except`.

January 10, 2022

* Cleaned up installation guide.
* Improved explanation of required and optional python packages.
* Made OpenCMP run without the optional packages installed, as long as their functionality is not called.
* Further cleaning up of directory structure.
* Changed pytests to run on a single core.
* Moved the very slow pytests behind an opt-in flag.
* Added support for running pytests in parallel.

December 14, 2021

* Cleaning up directory structure.
* Added ability to include.

December 12, 2021

* Added a tutorial showing how to use the Diffuse Interface Method to solve a 3D heat transfer problem.

December 5, 2021

* Undid changes to Stokes, they did not work properly when a transient solve was run.
* Updated requirements in the installation guide.

August 20, 2021

* Added the ability to also modeller controllers within the system (e.g. PID). Currently only a PID controller is implemented, MPC to come next.
* Simplified Stokes and MCINS by taking better advantage inheritance and moving more common code into INS.
* Changed the signature for the internal weak form-generating functions of INS, Stokes, and MCINS in order to make them congruent.
* Changed how Model.model_components, Model.model_local_error_components, Model.time_derivative_components, and Model.BCs are created in order to make it more clear where and how they need to be defined for custom models.
* Fixed some bugs with RK 222 and RK 232, thought they are still not working properly.
* Added more tests

August 13, 2021

* Added documentation, particularly notes for contributors on how to add new models and time discretization schemes.
* When specifying parameter values through an "IMPORT" the imported function now takes as arguments the model variable and parameter values at each time step in the time discretization scheme.

July 23, 2021

* Added ability for a model to have multiple weak forms. This is necessary for highly nonlinear models which solve each model variable separately (with different linearized forms of the weak form) during the overall solve at each time step.
* Added ability to specify parameter values in config functions from imported Python functions.
* Added rigid body rotation of phase fields for use in simulating impellers in chemical reactors.
* DIM simulations now save the phase field in addition to the usual model variables if specified in the main configuration file. Phase fields are saved to the "output_phi/" directory and converted to a .pvd file in said directory if specified.
* Fixed bug in CNLF so it now gives the expected second-order convergence for time step refinement.
* Configuration file names have changed to indicate the subdirectory in which they can be found. They are now "config", "bc_config", "ic_config", "model_config", "ref_sol_config", "dim_config", and "dim_bc_config".
* Removed the "Example Runs/" directory. All examples and tutorials can be found in "Examples/".

June 4, 2021

* Minor bug fixes.
* Added documentation including tutorials to walk through OpenCMP's functionality.

May 14, 2021

* Added the multicomponent INS model.
* Boundary condition values, model parameter values, etc are now saved for every step in the time discretization scheme. This fixes an implementation error that was causing suboptimal time step convergence rates for the higher-order time discretization schemes.
* The three-step adaptive time-stepping scheme is currently broken.
* Added Runge Kutta IMEX schemes, but they don't entirely work yet.

April 30, 2021

* Added option to use no preconditioner (specify preconditioner as "None").
* Several improvements to the code for the diffuse interface method.

    - To improve the fidelity of the phase field values outside of the expected range of [0,1] are now truncated to that range.
    - Added option to load :math:`\phi`, :math:`\underline{\nabla} \phi`, and :math:`\lvert \underline{\nabla} \phi \rvert` all from file instead of loading only :math:`\phi` from file and then generating the remaining phase fields from it.
    - Added option to generate structured triangular/tetrahedral meshes so adaptive mesh refinement can be used.

* Removed "pinned" boundary condition from the Poisson model. Use a Dirichlet boundary condition instead.
* Minor bug fixes and changes to/additional unit tests.

April 23, 2021

* Added multi-component INS model and examples.
* Changed the formatting of the pytest integration tests.
