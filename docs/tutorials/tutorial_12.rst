.. Contains the twelfth tutorial.
.. _tutorial_12:

Tutorial 12 - Simulating a Two-Fluid-Model Bubble Column (2D)
================================================================

The files for this tutorial can be found in ``examples/TFM``.

This tutorial demonstrates OpenCMP's transient ``TwoFluidModel`` using a
two-dimensional air--water bubble column.  It is a reduced-dimensional version
of the three-dimensional configuration studied by Fazeli, Rhebergen, and
Abukhdeir [1]_.  The equations, material properties, interphase-momentum
closures, and injection concept follow that work.

Governing Equations
-------------------

The continuous liquid phase and dispersed gas phase are treated as
interpenetrating continua.  Their volume fractions satisfy
:math:`\alpha_c+\alpha_d=1`.  Writing :math:`\alpha_d=1-\alpha_c`, the
constant-density phase mass balances can be expressed as

.. math::
   \partial_t\alpha_c
   +\nabla\cdot\bigl((\alpha_c-1)\boldsymbol{u}_d\bigr)
   &= -S_d, \\
   \nabla\cdot\bigl(\alpha_c\boldsymbol{u}_c
   +(1-\alpha_c)\boldsymbol{u}_d\bigr)
   &= S_d.

Here :math:`S_d` is nonzero only in the gas-injection region.  The interphase
momentum exchange is equal and opposite between the phases.  This example
includes drag, lift, virtual-mass, and laminar-dispersion contributions:

.. math::
   \boldsymbol{\mathcal{M}}'_d
   =\boldsymbol{F}_D+\boldsymbol{F}_L+\boldsymbol{F}_{VM}
    +\boldsymbol{F}_{LD}, \qquad
   \boldsymbol{\mathcal{M}}'_c=-\boldsymbol{\mathcal{M}}'_d.

The effect of the laminar-dispersion force is investigated in [1]_.

OpenCMP provides two canonical forms through ``canonical_form``.  They use the
same mass balances, pressure, gravity, and interphase momentum closures, but
differ in their treatment of molecular momentum fluxes.  The B-TFM formulation
is due to Brennen [2]_; its momentum equations in primitive form are

.. math::
   \partial_t\boldsymbol{u}_c
   +\boldsymbol{u}_c\cdot\nabla\boldsymbol{u}_c
   &= -\frac{\nabla p}{\rho_c}
      +\frac{1}{\alpha_c\rho_c}\nabla\cdot\boldsymbol{\tau}^{\mathrm{eff}}_c
      +\boldsymbol{g}
      -\frac{\boldsymbol{\mathcal{M}}'_d}{\alpha_c\rho_c}, \\
   \partial_t\boldsymbol{u}_d
   +\boldsymbol{u}_d\cdot\nabla\boldsymbol{u}_d
   &= -\frac{\nabla p}{\rho_d}
      +\boldsymbol{g}
      +\frac{\boldsymbol{\mathcal{M}}'_d}{\alpha_d\rho_d}.

The classical C-TFM formulation described by Ishii and Hibiki [3]_ instead
retains phase-weighted molecular stress terms in both phase momentum equations:

.. math::
   \partial_t\boldsymbol{u}_c
   +\boldsymbol{u}_c\cdot\nabla\boldsymbol{u}_c
   &= -\frac{\nabla p}{\rho_c}
      +\frac{1}{\alpha_c\rho_c}
       \nabla\cdot(\alpha_c\boldsymbol{\tau}^{\mathrm{eff}}_c)
      +\boldsymbol{g}
      -\frac{\boldsymbol{\mathcal{M}}'_d}{\alpha_c\rho_c}, \\
   \partial_t\boldsymbol{u}_d
   +\boldsymbol{u}_d\cdot\nabla\boldsymbol{u}_d
   &= -\frac{\nabla p}{\rho_d}
      +\frac{1}{\alpha_d\rho_d}
       \nabla\cdot(\alpha_d\boldsymbol{\tau}_d)
      +\boldsymbol{g}
      +\frac{\boldsymbol{\mathcal{M}}'_d}{\alpha_d\rho_d}.

The principal distinction is therefore the dispersed-phase molecular stress
:math:`\nabla\cdot(\alpha_d\boldsymbol{\tau}_d)`, which is present in C-TFM
and absent in B-TFM.  The continuous-phase stress is also averaged differently:
C-TFM uses :math:`\nabla\cdot(\alpha_c\boldsymbol{\tau}^{\mathrm{eff}}_c)`,
whereas B-TFM uses :math:`\nabla\cdot\boldsymbol{\tau}^{\mathrm{eff}}_c`.
For this tutorial, ``canonical_form = C-TFM`` selects the classical form.

Geometry and Injection Configuration
------------------------------------

The Gmsh geometry in ``2D_sample.msh`` is a :math:`0.20\,\mathrm{m}` wide by
:math:`0.45\,\mathrm{m}` high column.  A :math:`0.04\,\mathrm{m}` by
:math:`0.02\,\mathrm{m}` rectangular injection region is centred immediately
above the bottom boundary.  The physical surface named ``injection`` activates
the dispersed-phase mass and momentum sources; the remainder is named
``surface``.  Boundary curves are marked ``wall``, ``bottom``, and ``outlet``.

The model configuration supplies the source parameters::

   [INJECTION]
   region         = injection
   mass_flow_rate = 0.3
   velocity       = 0.2

Thus gas is introduced only in ``injection`` with an upward injection velocity
of :math:`0.2\,\mathrm{m/s}`.

The Main Configuration File
---------------------------

The velocity spaces are H(div)-conforming, while pressure and continuous-phase
volume fraction use discontinuous L2 spaces::

   [FINITE ELEMENT SPACE]
   elements = u_c     -> HDiv
              u_d     -> HDiv
              p       -> L2
              alpha_c -> L2
   interpolant_order = 3

   [DG]
   DG = True
   interior_penalty_coefficient = 10.0

The nonlinear problem is advanced with implicit Euler and Picard iteration::

   [SOLVER]
   linear_solver            = direct
   linearization_method     = Picard
   nonlinear_tolerance      = relative -> 1e-5
                              absolute -> 1e-5
   nonlinear_max_iterations = 10

   [TRANSIENT]
   transient  = True
   scheme     = implicit euler
   time_range = 0.0, 3.0
   dt         = 2e-3

``slope_limiter`` bounds the transported volume fraction, while
``diffusion_switch`` enables the configured artificial diffusion.  The TFM
section selects the canonical form and interphase closures::

   [OTHER]
   model            = TwoFluidModel
   slope_limiter    = True
   diffusion_switch = True

   [TFM]
   canonical_form = C-TFM
   IME = drag               -> Tomiyama
         lift               -> Tomiyama
         virtual_mass       -> ConstantCoefficient
         laminar_dispersion -> ConstantCoefficient
   lift_wall_deactivation = True
   lift_wall_boundaries   = wall|bottom

The Model Configuration File
----------------------------

The physical properties represent :math:`3\,\mathrm{mm}` air bubbles in water::

   [PARAMETERS]
   rho_c        = all -> 998.2
   rho_d        = all -> 1.204
   nu_c         = all -> 1.0038e-6
   nu_d         = all -> 1.5158e-5
   sigma_c      = all -> 0.072
   dp           = all -> 3e-3
   c_vm         = all -> 0.5
   cdis         = all -> 4.545
   d_artificial = all -> 1e-5

Boundary and Initial Conditions
-------------------------------

The liquid satisfies no slip on the side and bottom walls.  The dispersed
phase has a slip condition there, constraining its normal velocity while
leaving tangential motion free.  Both velocities have zero stress at the open
top, and ``ZERO_BACKFLOW`` supplies the outflow treatment for :math:`\alpha_c`::

   [DIRICHLET]
   u_c = wall   -> [0.0, 0.0]
         bottom -> [0.0, 0.0]

   [SLIP]
   u_d = wall|bottom

   [ZERO_STRESS]
   u_c = outlet
   u_d = outlet

   [ZERO_BACKFLOW]
   alpha_c = outlet|wall|bottom

The column initially contains quiescent liquid and no gas::

   [TwoFluidModel]
   u_c     = all -> [0.0, 0.0]
   u_d     = all -> [0.0, 0.0]
   alpha_c = all -> 1.0

Running and Visualizing the Simulation
--------------------------------------

From ``examples/TFM``, run:

.. code-block:: console

   python3 -m opencmp config

The supplied configuration currently has ``resume_from_previous = True`` and
``restart_from = LATEST``.  Keep these settings to continue an existing run;
disable ``resume_from_previous`` when starting from the initial conditions in a
clean output directory.  Open the generated output in ParaView to inspect
:math:`\alpha_c`, :math:`\boldsymbol{u}_c`, and :math:`\boldsymbol{u}_d`.

The following animation shows these three solution fields for the example:

.. raw:: html

   <video width="100%" controls muted loop playsinline>
     <source src="../_static/tfm_anim.mp4" type="video/mp4">
     Your browser does not support embedded MP4 video.
   </video>

.. [1] A. Fazeli, S. Rhebergen, and N. M. Abukhdeir, "Laminar dispersion
   force effects on two-fluid modelling and simulation of bubble column
   hydrodynamics," *International Journal of Multiphase Flow*, 105590, 2026.

.. [2] C. E. Brennen, *Fundamentals of Multiphase Flow*, Cambridge University
   Press, 2005.

.. [3] M. Ishii and T. Hibiki, *Thermo-Fluid Dynamics of Two-Phase Flow*,
   Springer Science & Business Media, 2010. doi:10.1007/978-1-4419-7985-8.
