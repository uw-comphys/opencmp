.. Notes on how to add new time discretization schemes to OpenCMP.
.. _adding_time_schemes:

Adding Time Discretization Schemes to OpenCMP
=============================================

The OpenCMP code base is designed such that new time discretization schemes can be added without significantly modifying the main code base. The time discretization schemes are defined in "time_integration_schemes.py", but the actual simulation run is controlled by a solver, the general hierarchy of which is shown below:

.. aafig::
   "Solver"
   |
   +-- "StationarySolver"
   |
   +-- "TransientMultiStepSolver"
   |       |
   |       +-- "BaseAdaptiveTransientMultiStepSolver"
   |               |
   |               +-- "AdaptiveTwoStep"
   |               |
   |               +-- "AdaptiveIMEX"
   |
   +-- "TransientRKSolver"
           |
           +-- "BaseAdaptiveTransientRKSolver"
                   |
                   +-- "AdaptiveThreeStep"
   
For more information on the differences between multi-step and Runge-Kutta time discretization schemes see :ref:`time_schemes`.

To add a new time discretization scheme, add it as a new function in "time_integration_schemes.py". Then determine whether it is a multi-step or Runge-Kutta scheme and modify the initialization of "solvers/base_solver.py" appropriately. Add the order of the scheme and its time step coefficients to :code:`scheme_order` and :code:`scheme_dt_coef` at the top of "solvers/base_solver.py". The order of the scheme is the number of previous time steps used by the scheme. For example, Crank-Nicolson uses only the one previous time step to solve for the next time step so has a scheme order of 1. The time step coefficients are only non-unity for Runge-Kutta schemes and indicate the portion of the full time step to use during each intermediate step. Fixed time step schemes can then simply be added to the :code:`_create_linear_and_bilinear_forms` method of :code:`TransientMultiStepSolver` or :code:`TransientRKSolver`. Finally, add the scheme to "solvers/misc.py".

Adaptive time-stepping schemes should inherit from either :code:`BaseAdaptiveTransientMultiStepSolver` or :code:`BaseAdaptiveTransientMultiStepSolver`. The exact implementation will vary depending on the scheme, but existing adaptive time-stepping schemes may provide some guide. The final class must be added to "solvers/adaptive_transient_solvers/__init__.py" and to "solvers/misc.py".


