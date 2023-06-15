.. Contains the seventh tutorial.
.. _tutorial_7:

Tutorial 7 - IMEX Schemes
=========================

The files for this tutorial can be found in "examples/tutorial_7".

Governing Equations
-------------------

This tutorial will solve the same governing equations as :ref:`tutorial_6`, but use an IMEX scheme to handle the nonlinearity in the incompressible Navier-Stokes equations. The governing equations are split up into nonlinear terms (in this case the convection term) which are treated explicitly, and linear terms which are treated implicitly. For more details on the implementation of IMEX schemes in OpenCMP see :ref:`time_schemes`.

The Main Configuration Files
----------------------------

There are still two main configuration files: "config_IC" for the Stokes solve and "config" for the incompressible Navier-Stokes solve. "config_IC" is identical to that used in :ref:`tutorial_7`.

"config" differs in the solver section. The linearization method is now "IMEX" and the Picard iteration parameters are no longer needed. ::

   [SOLVER]
   linear_solver = default
   preconditioner = default
   linearization_method = IMEX

The time discretization scheme must also be changed to an IMEX scheme. For this tutorial the Euler IMEX time discretization scheme is used, which is a combination of implicit and explicit Euler. ::

   [TRANSIENT]
   transient = True
   scheme = euler IMEX
   time_range = 0.0, 1.0
   dt = 1e-2

Running the Simulation
----------------------

The simulation can be run from the command line; within the directory examples/tutorial_7/ execute :code:`python3 -m opencmp config`.

The simulation can be run from the command line; within the directory "examples/tutorial_7/::

1) Run the Stokes solve by calling :code:`python3 -m opencmp config_IC`
2) Run the incompressible Navier-Stokes solve by calling :code:`python3 -m opencmp config`.

As usual, the progress of the transient simulation can be tracked from the print outs at each time step. Once the simulation has finished the results can be visualized in ParaView. They should be identical to those obtained in :ref:`tutorial_7`.
