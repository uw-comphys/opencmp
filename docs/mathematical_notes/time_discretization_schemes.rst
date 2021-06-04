.. Notes on the various time discretization schemes.
.. _time_schemes:

Time Discretization Schemes
===========================

The time-stepping schemes used in OpenCMP can be split between those used with fixed time steps and those used for adaptive time-stepping. Each category can be further split into single-step, multi-step and Runge-Kutta schemes. Finally, these further categories can be split by the type of linearization done. At present, the only nonlinear model is incompressible Navier-Stokes, which can be linearized either as the Oseen equation or by using an IMEX time-stepping scheme.

Definitions
-----------

*Single-Step Schemes:* Use only the values at the previous time step.

*Multi-Step Schemes:* Use the values from several previous time steps (require storing information from several previous time steps).

*Runge-Kutta Schemes:* Use only the values at the previous time step, but take one or more intermediate steps within a time step.

*IMEX Schemes:* Split a nonlinear equation into the nonlinear terms :math:`f(u)` and the linear terms :math:`g(u)`. The nonlinear terms are solved explicitly to avoid the need for a nonlinear solver while the linear terms are still solved implicitly. Typically the linear terms are very stiff and would severely limit the time step if solved explicitly. 

Notation
--------

For the remainder of this document the governing equation to be solved will be taken as 

.. math::
   \frac{\partial u}{\partial t} = F(u)

When discussing IMEX schemes :math:`F(u)` can be split as

.. math::
   F(u) = f(u) + g(u)

where, as above, :math:`f(u)` represents the nonlinear terms and :math:`g(u)` represents the linear terms.

The variable values at each time step will be denoted by a superscript. For example, :math:`u^{n+1}` is the value of :math:`u` to be solved for at the next time step, while :math:`u^n` is the known value of :math:`u` from the previous time step.

The time step will be denoted as :math:`\Delta t`, again with a superscript if it changes at each time step as with an adaptive time-stepping scheme.


Fixed Time Step Schemes
-----------------------

The following time-stepping schemes have been implemented for use with a fixed time step.

Implicit Euler
**************

Implicit Euler is a first-order accurate single-step scheme.

.. math::
   \frac{u^{n+1} - u^n}{\Delta t} = F(u^{n+1})

Explicit Euler
**************

Explicit Euler is a first-order accurate single-step scheme.

.. math::
   \frac{u^{n+1} - u^n}{\Delta t} = F(u^n)

Crank-Nicolson (Trapezoidal Rule)
*********************************

Crank-Nicolson is a second-order accurate single-step scheme.

.. math::
   \frac{u^{n+1} - u^n}{\Delta t} = \frac{1}{2} \left[ F(u^{n+1}) + F(u^n) \right]

First-Order IMEX (Euler IMEX)
*****************************

Euler IMEX is a first-order accurate single-step IMEX scheme. This could also be considered a first-order SBDF scheme.

.. math::
   \frac{u^{n+1} - u^n}{\Delta t} = f(u^n) + g(u^{n+1})

CNLF (Crank-Nicolson Leap-Frog)
*******************************

CNLF is a second-order accurate multi-step IMEX scheme.

.. math::
   \frac{u^{n+1} - u^{n-1}}{2\Delta t} = f(u^n) + \frac{1}{2} \left[ g(u^{n+1}) + g(u^{n-1}) \right]

SBDF (Semi-Implicit Backwards Difference)
*****************************************

SBDF is a third-order accurate multi-step IMEX scheme. SBDF can also refer to a family of IMEX schemes of various orders, but only the third-order scheme has been implemented in OpenCMP.

.. math::
   \frac{1}{\Delta t} \left[ \frac{11}{6} u^{n+1} - 3 u^n + \frac{3}{2} u^{n-1} - \frac{1}{3} u^{n-2} \right] = 3f(u^n) - 3f(u^{n-1}) + f(u^{n-2}) + g(u^{n+1})

(2,3,2)
*******

(2,3,2) is a second-order accurate Runge-Kutta IMEX scheme. 

First Intermediate Step: :math:`\Delta t^1 = \gamma \Delta t`

.. math::
   \frac{u^1 - u^n}{\Delta t} = \gamma \left[ f(u^n) + g(u^1) \right]

Second Intermediate Step: :math:`\Delta t^2 = \Delta t`

.. math::
   \frac{u^2 - u^n}{\Delta t} = \delta f(u^n) + (1-\delta) f(u^1) + (1-\gamma) g(u^1) + \gamma g(u^2)

Final Solve for :math:`u^{n+1}`:

.. math::
   \frac{u^{n+1} - u^n}{\Delta t} = (1-\gamma) f(u^1) + \gamma f(u^2) + (1-\gamma) g(u^1) + \gamma g(u^{n+1})

The implemented scheme uses :math:`\gamma = \frac{2 - \sqrt{2}}{2}` and :math:`\delta = \frac{-2\sqrt{2}}{3}`.

(2,2,2)
*******

(2,2,2) is a second-order accurate Runge-Kutta IMEX scheme. 

Intermediate Step: :math:`\Delta t^1 = \gamma \Delta t`

.. math::
   \frac{u^1 - u^n}{\Delta t} = \gamma \left[ f(u^n) + g(u^1) \right]

Final Solve for :math:`u^{n+1}`:

.. math::
   \frac{u^{n+1} - u^n}{\Delta t} = \delta f(u^n) + (1-\delta) f(u^1) + (1-\gamma) g(u^1) + \gamma g(u^{n+1})

The implemented scheme uses :math:`\gamma = \frac{2-\sqrt{2}}{2}` and :math:`\delta = 1-\frac{1}{2 - \sqrt{2}}`.

Adaptive Time-Stepping Schemes
------------------------------

The following adaptive time-stepping schemes have been implemented.

Adaptive Two Step
*****************

This scheme compares a Crank-Nicolson solve to an implicit Euler solve in order to constrain local error. If the time step is accepted the implicit Euler solution is taken as the time step's solution in order to preserve stability. 

Adaptive Three Step
*******************

This scheme compares one solve of implicit Euler with two solves of implicit Euler each using :math:`\frac{1}{2} \Delta t` in order to constrain local error. If the time step is accepted the second implicit Euler solve using :math:`\frac{1}{2}\Delta t` is taken as the time step's solution for highest accuracy.

Adaptive IMEX
*************

This scheme defines several new operators.

.. math::
   w^n &= \frac{\Delta t^n}{\Delta t^{n-1}} \\
   E(u^n) &= (1 + w^n) u^n - w^n u^{n-1}

It also uses the explicitly skew-symmetric form of the convection term in incompressible Navier-Stokes

.. math::
   \frac{1}{2} \left[ \underline{u} \cdot \underline{\nabla} \underline{u} + \underline{\nabla} \cdot \left( \underline{u} \underline{u} \right) \right]

This is not used in the OpenCMP implementation because it was introducing large errors when :math:`\underline{\nabla} \cdot \underline{u}` was not exactly zero.

Predictor:

.. math::
   \frac{u^{n+1}_p - u^n}{\Delta t} = f(E(u^n)) + g(u^{n+1})

Corrector:

.. math::
   u^{n+1}_c = u^{n+1}_p - \frac{w^n}{2w^n + 1} \left[ u^{n+1}_p - E(u^n) \right]

There are also two local error estimators.

.. math::
   e_1 &= \lvert \lvert u_c^{n+1} - u_p^{n+1} \rvert \rvert \\
   e_2 &= \lvert \lvert \frac{w^{n-1} w^n (1 + w^n)}{1 + 2w^n + w^{n-1} (1 + 4w^n + 3(w^n)^2)} \\
   &\left[ u^{n+1}_c - \frac{(1+w^n)(1+w^{n-1}(1+w^n))}{1+w^{n-1}} u^n + w^n(1+w^{n-1}(1+w^n)) u^{n-1} - \frac{(w^{n-1})^2 w^n(1+w^n)}{1+w^{n-1}} u^{n-2} \right] \rvert \rvert

A time step's solution is accepted if either local error estimate is below the user-specified tolerance. The next time step is chosen using the highest local error estimate that satisfies the tolerance. The time step's solution is taken as :math:`u_p^{n+1}` if :math:`e_1` is the highest still-acceptable local error estimate or :math:`u_c^{n+1}` if :math:`e_2` is the highest still-acceptable local error estimate.

References
----------

First-order IMEX, CNLF, and SBDF are taken from `Ascher1995 <https://doi.org/10.1137/0732037>`_ *Implicit-Explicit Methods for Time-Dependent PDEs*

(2,3,2) and (2,2,2) are taken from `Ascher1997 <https://doi.org/10.1016/S0168-9274(97)00056-1>`_ *Implicit-Explicit Runge-Kutta Methods for Time-Dependent Partial Differential Equations*

Adaptive IMEX is taken from `DeCaria2021 <https://doi.org/10.1016/j.cma.2020.113661>`_ *An Embedded Variable Step IMEX Scheme for the Incompressible Navier-Stokes Equations* 
