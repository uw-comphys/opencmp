.. Contains the first tutorial.
.. _tutorial_1:

Tutorial 1 - Introducing the User Interface
===========================================

The files for this tutorial can be found in "Examples/tutorial_1".

Directory Structure
-------------------

Drawing inspiration from packages like OpenFOAM and SU2, OpenCMP's user interface is organized around configuration files and the command line. Each simulation requires its own directory to hold its configuration files and outputs. This is known as the run directory or run_dir. The standard layout of this directory is shown below.

.. aafig::
   simulation
   |
   +-- config
   |
   +-- "bc_dir"
   |      |
   |      +-- "bc_config"
   |
   +-- "ic_dir"
   |      |
   |      +-- "ic_config"
   |
   +-- "model_dir"
   |      |
   |      +-- "model_config"
   |
   +-- "ref_sol_dir"
   |      |
   |      +-- "ref_sol_config"
   |
   +-- output

The main directory and each subdirectory contain a configuration file - "< >_config". These are plaintext files that specify the simulation parameters and run conditions.

The Main Directory
------------------

The configuration file in the main directory holds general information about the simulation including which model, mesh, finite elements, and solver should be used. It also holds information about how the simulation should be run such as the level of detail in the output messages and the amount of multi-threading to use.

The Boundary Condition Subdirectory
-----------------------------------

The "bc_dir" subdirectory holds information about the boundary conditions. Its configuration file specifies the type and value of each boundary condition. This subdirectory can also hold files containing boundary condition data if a boundary condition value is to be loaded from file instead of given in closed form.

The Initial Condition Subdirectory
----------------------------------

The "ic_dir" subdirectory holds information about the initial conditions. Its configuration file specifies the value of the initial condition for each model variable. Like "bc_dir", "ic_dir" may contain additional files from which the initial condition data is loaded during the simulation.

The Model Subdirectory
----------------------

The "model_dir" subdirectory holds information about model parameters and model functions. Its configuration file specifies the values of any model parameters or functions for each model variable and the subdirectory may hold additional data files to be loaded during the simulation.

The Error Analysis Subdirectory
-------------------------------

The "ref_sol_dir" subdirectory holds information about the error analysis to be conducted on the final simulation result. Its configuration file specifies what error metrics should be computed during post-processing. This configuration file also contains the reference solutions the results should be compared against, either in closed form or as references to other files in the subdirectory that are loaded during post-processing.
 
The Output Subdirectory
-----------------------

The output subdirectory holds any saved simulation results. It doesn't need to be created before running the simulation, it will be generated automatically if results should be saved to file.
