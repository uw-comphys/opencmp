.. A reference for all of the options that can be specified in a configuration file.
.. _example_config:

Configuration File Options Guide
================================

The following parameters can be specified in the configuration files. Some parameters are specific to certain models. Many parameters have default values and do not need to always be explicitly specified.

Main Configuration File
-----------------------

This is the main configuration file for the simulation run. It is kept in the run directory.

+---------------+------------------------------+--------------------+----------------+----------------------------+
| Header        | Parameter                    | Expected Type      | Default Value  | Description                |
+===============+==============================+====================+================+============================+
| MESH          | filename                     | filepath           |                | The path to the mesh file. |
|               |                              |                    |                | This can be a .vol file or |
|               |                              |                    |                | a .msh file.               |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | curved_elements              | True/False         | False          | Whether to use curved      |
|               |                              |                    |                | elements to better         |
|               |                              |                    |                | approximate the domain     |
|               |                              |                    |                | boundary.                  |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| DIM           | diffuse_interface_method     | True/False         | False          | Whether to use the diffuse |
|               |                              |                    |                | interface method           |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | dim_dir                      | filepath           |                | The path to the DIM        |
|               |                              |                    |                | directory.                 | 
+---------------+------------------------------+--------------------+----------------+----------------------------+
| FINITE        | elements                     | var -> name        |                | The type of finite element |
| ELEMENT       |                              |                    |                | space (name) to use for    |
| SPACE         |                              |                    |                | each model variable (var). |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | interpolant_order            | integer            |                | The highest order of       |
|               |                              |                    |                | interpolant to use.        |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | no_constrained_dofs          | True/False         | False          | Whether to solve on all    |
|               |                              |                    |                | DOFs or just free DOFs.    |
|               |                              |                    |                | Only set to True if using  |
|               |                              |                    |                | RT finite element spaces.  |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| DG            | DG                           | True/False         | False          | Whether to use             |
|               |                              |                    |                | Discontinuous Galerkin.    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | interior_penalty_coefficient | number             | 10             | Coefficient for interior   |
|               |                              |                    |                | penalty method DG.         |
|               |                              |                    |                | C in C*n^2/h.              |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| SOLVER        | solver                       | name               | default        | The type of solver to use. |
|               |                              |                    |                | Options are direct, CG,    |
|               |                              |                    |                | MinRes, GMRes, Richardson. |
|               |                              |                    |                | Defaults to direct.        |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | preconditioner               | name               | default        | The type of preconditioner |
|               |                              |                    |                | to use. Options are None,  |
|               |                              |                    |                | local, direct, multigrid,  |
|               |                              |                    |                | h1amg, bddc. Defaults to   |
|               |                              |                    |                | local.                     |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | solver_tolerance             | number             | 1e-8           | Stopping tolerance for an  |
|               |                              |                    |                | iterative solve.           |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | solver_max_iterations        | integer            | 100            | Maximum number of          |
|               |                              |                    |                | iterations for an          |
|               |                              |                    |                | iterative solve.           |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | linearization_method         | name               | Oseen          | Method for linearizing a   | 
|               |                              |                    |                | nonlinear model. Options   |
|               |                              |                    |                | are Oseen or IMEX.         |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | nonlinear_tolerance          | relative -> number | 0 for          | Tolerance between          |
|               |                              +--------------------+ absolute       | successive iterations when |
|               |                              | absolute -> number | tolerance      | linearizing a nonlinear    |
|               |                              |                    |                | model as the Oseen method. |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | nonlinear_max_iterations     | integer            | 10             | Maximum number of          |
|               |                              |                    |                | iterations when            |
|               |                              |                    |                | linearizing a nonlinear    |
|               |                              |                    |                | model as the Oseen method. |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| TRANSIENT     | transient                    | True/False         | False          | Whether the solve is       |
|               |                              |                    |                | transient or stationary.   |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | scheme                       | name               | implicit euler | The time integration       |
|               |                              |                    |                | scheme to use. Options are |
|               |                              |                    |                | implicit euler, explicit   |
|               |                              |                    |                | euler, crank nicolson,     |
|               |                              |                    |                | euler IMEX, CNLF, SBDF,    |
|               |                              |                    |                | RK 222, RK 232, adaptive   |
|               |                              |                    |                | two step, adaptive three   |
|               |                              |                    |                | step, adaptive IMEX.       |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | time_range                   | number, number     | 0, 5           | The start and end time.    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | dt                           | number             | 0.001          | The time step or the       |
|               |                              |                    |                | initial time step for an   |
|               |                              |                    |                | adaptive time-stepping     |
|               |                              |                    |                | scheme.                    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | dt_tolerance                 | relative -> number | 0 for          | Local error tolerance for  |
|               |                              +--------------------+ absolute       | adaptive time-stepping     |
|               |                              | absolute -> number | tolerance      | schemes.                   |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | dt_range                     | number, number     | 1e-12, 0.1     | Minimum and maximum        |
|               |                              |                    |                | allowed time step for an   |
|               |                              |                    |                | adaptive time-stepping     |
|               |                              |                    |                | scheme.                    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | maximum_rejected_solves      | integer            | 1000           | Maximum number of rejected |
|               |                              |                    |                | time steps before an       |
|               |                              |                    |                | adaptive time-stepping     |
|               |                              |                    |                | scheme quits.              |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| ERROR         | check_error                  | True/False         | False          | If True computes the error |
| ANALYSIS      |                              |                    |                | of the final result        |
|               |                              |                    |                | relative to a specified    |
|               |                              |                    |                | reference solution.        |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | check_error_every_timestep   | True/False         | False          | If True computes the error |
|               |                              |                    |                | relative to a specified    |
|               |                              |                    |                | reference solution after   |
|               |                              |                    |                | every time step.           |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | save_error_every_timestep    | True/False         | False          | If True computes the error |
|               |                              |                    |                | after every time step and  |
|               |                              |                    |                | saves it to file in the    |
|               |                              |                    |                | "output" subdirectory.     |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | convergence_test             | h -> vars/False    | False for both | If vars is given as a      |
|               |                              +--------------------+                | variable name or list of   |
|               |                              | p -> vars/False    |                | variable names, that type  |
|               |                              |                    |                | of convergence test        |
|               |                              |                    |                | (h or p) is run on those   |
|               |                              |                    |                | variables.                 |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | error_average                | var, var...        | Nothing        | Which variables should be  |
|               |                              |                    |                | biased to a zero mean      |
|               |                              |                    |                | before computing error.    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | num_refinements              | integer            | 4              | The number of refinement   |
|               |                              |                    |                | steps taken by the         |
|               |                              |                    |                | convergence test(s).       |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| VISUALIZATION | save_to_file                 | True/False         | False          | Whether to save results to |
|               |                              |                    |                | file.                      |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | save_type                    | name               | .sol           | The file format to save    |
|               |                              |                    |                | to. Options are .sol or    |
|               |                              |                    |                | .vtu. Choosing .vtu also   |
|               |                              |                    |                | produces a .pvd with all   |
|               |                              |                    |                | of the .vtu files from     |
|               |                              |                    |                | each saved time step.      |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | save_frequency               | number, numit/time | 1, numit       | How often to save results. |
|               |                              |                    |                | The numit option specifies |
|               |                              |                    |                | saving every certain       |
|               |                              |                    |                | number of time steps (ex:  |
|               |                              |                    |                | 1, numit saves after each  |
|               |                              |                    |                | time step). The time       |
|               |                              |                    |                | option specifies saving    |
|               |                              |                    |                | at certain time intervals  |
|               |                              |                    |                | (ex: 0.1, time saves after |
|               |                              |                    |                | every additional 0.1s).    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | subdivision                  | integer            | the specified  | The interpolatation level  |
|               |                              |                    | interpolant    | if saving to .vtu.         |
|               |                              |                    | order          |                            |
+---------------+------------------------------+--------------------+----------------+----------------------------+
| OTHER         | num_threads                  | integer            | 4              | The number of threads to   |
|               |                              |                    |                | run the simulation on.     |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | model                        | name               |                | The model to simulate.     |
|               |                              |                    |                | Options are Poisson,       |
|               |                              |                    |                | Stokes, INS,               |
|               |                              |                    |                | MultiComponentINS.         |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | component_names              | name, name...      |                | Names of additional model  |
|               |                              |                    |                | variables for              |
|               |                              |                    |                | multicomponent flow. These |
|               |                              |                    |                | correspond to the various  |
|               |                              |                    |                | solutes present in the     |
|               |                              |                    |                | mixture.                   |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | component_in_time_deriv      | var -> True/False  |                | Whether each additional    |
|               |                              |                    |                | model variable has a time  |
|               |                              |                    |                | derivative.                |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | component_in_error_calc      | var -> True/False  |                | Whether each additional    |
|               |                              |                    |                | model variable should be   |
|               |                              |                    |                | included in the local      |
|               |                              |                    |                | error estimation if using  |
|               |                              |                    |                | an adaptive time-stepping  |
|               |                              |                    |                | scheme.                    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | velocity_fixed               | True/False         | False          | Only for multicomponent    |
|               |                              |                    |                | flow model. Whether or not |
|               |                              |                    |                | to solve the fluid flow at |
|               |                              |                    |                | each timestep, or to keep  |
|               |                              |                    |                | it fixed. Enable if the    |
|               |                              |                    |                | velocity does not need to  |
|               |                              |                    |                | be solved as it results in |
|               |                              |                    |                | significantly faster       |
|               |                              |                    |                | solves, and lower memory   |
|               |                              |                    |                | usage.                     |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | parameter_names              | name, name...      | Nothing        | Names of parameters that   |
|               |                              |                    |                | will be used as additional |
|               |                              |                    |                | variables in the model.    |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | run_dir                      | filepath           |                | Path to the main directory |
|               |                              |                    |                | for the simulation.        |
|               +------------------------------+--------------------+----------------+----------------------------+
|               | messaging_level              | integer            | 0              | The level of messaging to  |
|               |                              |                    |                | display during the         |
|               |                              |                    |                | simulation. Higher values  |
|               |                              |                    |                | increase the amount of     |
|               |                              |                    |                | information shown.         |
+---------------+------------------------------+--------------------+----------------+----------------------------+

Boundary Condition Configuration File
-------------------------------------

This configuration file holds information about the boundary conditions and is kept in the boundary condition directory. 

The headers correspond to the different types of boundary conditions. Then, for each type of boundary condition, the variable to apply the boundary condition to, the mesh marker to apply the boundary condition at, and the value of the boundary condition are specified as a multi-level parameter.

Below is an example of a simulation with a Dirichlet boundary condition applied to the model variable "u" at the "left" mesh marker and a Neumann boundary condition applied to the same model variable at the "right" mesh marker. ::

   [DIRICHLET]
   u = left -> sin(4*x)
   
   [NEUMANN]
   u = right -> -11.5
   
One special case is stress boundary conditions, which do not need to have a model variable specified. Instead, the model variable should be left as "stress". ::

   [STRESS]
   stress = left -> [5, 2*t]
   
The other special case is no-tangential-flow boundary conditions, which do not need to have a model variable specified and do not need to have a value specified. Instead, the model variable should be left as "parallel" and the value can be any value. ::

   [PARALLEL]
   parallel = left -> 1


Initial Condition Configuration File
------------------------------------

This configuration file holds information about the initial conditions and is kept in the initial condition directory.

The headers correspond to the models that will use the given initial conditions. For example, if an INS solve is being initialized by a Stokes solve there would be a STOKES header for the Stokes initial condition and an INS header for the INS initial condition.

For each model, an initial condition must be specified for each model variable. This can be done for each model separately. Alternatively, if the initial condition is being loaded from file one file can be used for all of the model variables. Different initial conditions can also be specified on different regions of the mesh. 

Below is an example where the initial condition is being specified separately for each model variable. For "u" the same initial condition is used over the entire mesh, while for "p" different regions of the mesh are initialized differently. ::

   [STOKES]
   u = all -> [y*(1 - y), 0.0]
   p = left_half  -> 5
       right_half -> 10
       
Here is an example where both model variables have initial conditions specified in the same file. ::

   [STOKES]
   all = all -> ic_file.sol

Model Configuration File
------------------------

This configuration file holds information about the model parameters and model functions. It is kept in the model directory.

There are two sections. PARAMETERS holds information about model parameters like kinematic viscosity or the diffusion coefficient. FUNCTIONS holds information about model functions like the source terms. Within both sections, each parameter must be specified as a multi-level parameter with information about which model variables the parameter applies to and the value of the parameter for each model variable.

Below is an example for multicomponent flow. The model only has one single kinematic viscosity, but each solute has its own diffusion coefficient. Sources terms are specified for the velocity and both solutes. ::

   [PARAMETERS]
   kinematic_viscosity = all -> 0.1
   diffusion_coefficients = a -> 1e-4
                            b -> 5e-3
   
   [FUNCTIONS]
   source = u -> [0, 0]
            a -> 0.1
            b -> -0.1

Reference Solution Configuration File
-------------------------------------

This configuration file holds information about the error analysis of the simulation results. It is kept in the reference solution directory.

The REFERENCE SOLUTIONS section holds the reference "exact" solutions. The reference solution for each model variable can be loaded from a single file. Alternatively, reference solutions can be specified for only some of the model variables either in closed form or to be loaded from file.

Below is an example where one single file holds the full reference solution. ::

   [REFERENCE SOLUTIONS]
   all = ref_sol_file.sol
   
Here is an example where reference solutions are specified separately for each model variable. ::

   [REFERENCE SOLUTIONS]
   u = [y*(1 - y), 0.0]
   p = ref_sol_p_file.sol
   
The METRICS section holds information about what errors to compute. Options are L1_norm, L2_norm, Linfinity_norm, divergence, facet_jumps, and surface_traction. These metrics can be computed for any or all of the model variables. However, if any of the norms are to be computed a reference solution must be given for the relevant model variable. For surface_traction, pass a list of mesh surface markers to compute on instead of a list of model variables.

Below is an example where a reference solution is given for both model variables and used to compute the L2 norm in error for the final simulation results. ::

   [REFERENCE SOLUTIONS]
   all = ref_sol_file.sol
   
   [METRICS]
   L2_norm = u, p 

Diffuse Interface Configuration Files
-------------------------------------

These configuration files hold information about the diffuse interface method and are kept in the appropriate subdirectories of the diffuse interface directory.

Main Diffuse Interface Configuration File
*****************************************

+------------+-----------------------------+--------------------------+----------------+----------------------------+
| Header     | Parameter                   | Expected Type            | Default Value  | Description                |
+============+=============================+==========================+================+============================+
| DIM        | mesh_dimension              | integer                  | 2              | The mesh dimension.        |
|            |                             |                          |                | Options are 2 or 3.        |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | num_mesh_elements           | x -> integer             | 59             | The number of mesh         |
|            |                             +--------------------------+----------------+ elements along each        |
|            |                             | y -> integer             | 59             | dimension. z is only       |
|            |                             +--------------------------+----------------+ necessary in 3D.           |
|            |                             | z -> integer             | 59             |                            |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | num_phi_mesh_elements       | x -> integer             |                | Similar to                 |
|            |                             +--------------------------+                | num_mesh_elements, but     |
|            |                             | y -> integer             |                | used if the phase fields   |
|            |                             +--------------------------+                | should be generated on a   |
|            |                             | z -> integer             |                | finer mesh than the        |
|            |                             |                          |                | simulation mesh and then   |
|            |                             |                          |                | be interpolated onto the   |
|            |                             |                          |                | simulation mesh.           |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | mesh_scale                  | x -> number              | 1              | The absolute extent of the |
|            |                             +--------------------------+----------------+ mesh along each dimension. |
|            |                             | y -> number              | 1              | z is only necessary in 3D. |
|            |                             +--------------------------+----------------+                            |
|            |                             | z -> number              | 1              |                            |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | mesh_offset                 | x -> number              | 0              | Centers the mesh along     |
|            |                             +--------------------------+----------------+ each dimension. z is only  |
|            |                             | y -> number              | 0              | necessary in 3D.           |
|            |                             +--------------------------+----------------+                            |
|            |                             | z -> number              | 0              |                            |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | quad_mesh                   | True/False               | True           | Whether a structured       |
|            |                             |                          |                | quad/hex mesh or a         |
|            |                             |                          |                | structured triangle/tet    |
|            |                             |                          |                | mesh should be used.       |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | interface_width_parameter   | number                   | 1e-5           | Controls the diffuseness   |
|            |                             |                          |                | of the diffuse interface.  |
+------------+-----------------------------+--------------------------+----------------+----------------------------+
| PHASE      | load_method                 | name                     |                | Specifies how to obtain    |
| FIELDS     |                             |                          |                | the phase fields. Options  |
|            |                             |                          |                | are file (the phase fields |
|            |                             |                          |                | are loaded from file),     |
|            |                             |                          |                | generate (the phase fields |
|            |                             |                          |                | are generated from a .stl  |
|            |                             |                          |                | file), or combine (the     |
|            |                             |                          |                | phase fields are generated |
|            |                             |                          |                | by combining multiple .stl |
|            |                             |                          |                | files).                    |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | stl_filename                | filepath or              |                | The path to the .stl file  | 
|            |                             | name -> filepath         |                | (or multiple .stl files)   |
|            |                             |                          |                | used to generate the phase |
|            |                             |                          |                | fields. Only necessary if  |
|            |                             |                          |                | "load_method" is           |
|            |                             |                          |                | "generate" or "combine".   |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | phase_field_filename        | phi -> filepath          |                | The path to the files      |
|            |                             +--------------------------+                | containing the phase       |
|            |                             | grad_phi -> filepath     |                | fields. Only necessary if  |
|            |                             +--------------------------+                | "load_method" is "file".   |
|            |                             | mag_grad_phi -> filepath |                | Only phi needs to be       |
|            |                             |                          |                | specified. grad_phi and    |
|            |                             |                          |                | mag_grad_phi will be       |
|            |                             |                          |                | generated from phi if they |
|            |                             |                          |                | are not given.             |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | invert_phi                  | True/False               | False          | Whether to invert the      |
|            |                             |                          |                | phase field after          |
|            |                             |                          |                | generating it from a .stl  |
|            |                             |                          |                | file.                      |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | save_to_file                | True/False               | True           | Whether to save the phase  |
|            |                             |                          |                | fields to file.            |
+------------+-----------------------------+--------------------------+----------------+----------------------------+
| DIM        | multiple_bcs                | True/False               | False          | Whether or not multiple    |
| BOUNDARY   |                             |                          |                | different boundary         |
| CONDITIONS |                             |                          |                | conditions should be       |
|            |                             |                          |                | applied to different       |
|            |                             |                          |                | regions of the interface.  |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | remainder                   | True/False               | False          | If True, after splitting   |
|            |                             |                          |                | the interface into         |
|            |                             |                          |                | multiple regions any       |
|            |                             |                          |                | remaining parts are        |
|            |                             |                          |                | assigned to the same       |
|            |                             |                          |                | additional region.         |
|            +-----------------------------+--------------------------+----------------+----------------------------+
|            | overlap_interface_parameter | number                   | -1             | If positive, controls how  |
|            |                             |                          |                | much different interface   |
|            |                             |                          |                | boundary conditions        |
|            |                             |                          |                | diffuse into each other.   |
|            |                             |                          |                | If negative, there is a    |
|            |                             |                          |                | sharp transition between   |
|            |                             |                          |                | boundary conditions.       |
+------------+-----------------------------+--------------------------+----------------+----------------------------+
| RIGID BODY | rotation_speed              | number, number           | 1, 0.25        | The final rotation speed   |
| MOTION     |                             |                          |                | (RPS) and the time taken   |
|            |                             |                          |                | to ramp from zero to the   |
|            |                             |                          |                | final rotation speed (ex:  |
|            |                             |                          |                | the defaults ramp from     |
|            |                             |                          |                | 0 RPS to 1 RPS in 0.25s    |
+------------+-----------------------------+--------------------------+----------------+----------------------------+

.. note:: The DIM section parameters only need to be specified if the phase fields are to be generated from .stl files or if the phase fields are to undergo rigid body motion.

.. note:: Currently rigid body motion of phase fields is only implemented as rotation about the z-axis centered on the origin for use in simulating impellers in stirred tank reactors.

Diffuse Interface Boundary Condition Configuration File
*******************************************************

The diffuse interface boundary condition configuration file has the same form as a standard boundary condition configuration file, with two additional sections if multiple boundary conditions are to be applied to the diffuse interface.

The VERTICES section holds information about the boundaries of the various different diffuse interface regions with different boundary conditions. For each different region, the bounding vertices of said region are specified either as a list of coordinates in counterclockwise order (2D) or a .stl file that maps the region (3D). 

The CENTROIDS section holds information about the centroids to use when splitting the diffuse interface into different regions. For each different region, the centroid of that region is specified either as a list of coordinates or not specified (set to "None").

Below is an example for a diffuse interface simulation with two different Dirichlet boundary conditions on different sections of the interface. This is a 2D example so the bounding vertices of the regions are specified as lists of coordinates. Only one region has a centroid specified. ::

   [VERTICES]
   top = <1.0, 0.0>, <-1.0, 0.0>
   bottom = <-1.0, 0.0>, <1.0, 0.0>
   
   [CENTROIDS]
   top = <0.0, 0.5>
   bottom = None
   
   [DIRICHLET]
   u = top    -> 5
       bottom -> 10

.. note:: The diffuse interface boundary condition configuration file specifies only the boundary conditions at the diffuse interface. Any conformal boundary conditions are still specified in the main boundary condition configuration file.

