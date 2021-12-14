.. A short script to provide an example NGSolve simulation workflow without all the overhead OpenCMP requires.
.. _example_code:

Basic Simulation Code
=====================

OpenCMP includes large amounts of auxilliary code to handle interactions with the user interface and the various post-processing capabilities. As such, it is recommended that new models be developed and debugged as short stand-alone scripts before being incorporated into OpenCMP. 

The following code solves the incompressible Navier-Stokes equations using IMEX-style linearization and a first-order IMEX time discretization. It can be used as an example and a starting point for further model development.

.. code:: python
   
   # Import the necessary NGSolve/Netgen modules.
   import ngsolve as ngs
   from netgen.geom2d import SplineGeometry
   
   # Generate the mesh.
   geo = SplineGeometry()
   geo.AddRectangle((0,0), (2,0.41), bcs=['wall', 'outlet', 'wall', 'inlet'])
   geo.AddCircle((0.2,0.2), r=0.05, leftdomain=0, rightdomain=1, bc='cyl')
   mesh = ngs.Mesh(geo.GenerateMesh(maxh=0.05))
   mesh.Curve(3)

   # Define the time variables.
   t_0 = 0.0
   t_end = 1.0
   dt = 0.001

   t_param = ngs.Parameter(t_0)
   dt_param = ngs.Parameter(dt)

   # Various problem values.
   kv = 1e-3
   interp_ord = 3
   u_inlet = ngs.CoefficientFunction((6 * ngs.y * (0.41 - ngs.y) / (0.41 * 0.41), 0.0))
   h_outlet = ngs.CoefficientFunction((0.0, 0.0))
   f = ngs.CoefficientFunction((0.0, 0.0))
   n = ngs.specialcf.normal(mesh.dim)

   # Define the finite element space.
   V = ngs.VectorH1(mesh, order=interp_ord, dirichlet='wall|inlet|cyl')
   Q = ngs.H1(mesh, order=interp_ord - 1)
   X = ngs.FESpace([V,Q])

   # Define the trial and test functions.
   u, p = X.TrialFunction()
   v, q = X.TestFunction()

   # Define gridfunctions to hold the current solution and previous time step.
   gfu = ngs.GridFunction(X)
   gfu_0 = ngs.GridFunction(X)

   # Obtain the initial condition from a Stokes solve.
   a_st = ngs.BilinearForm(X)
   a_st += (kv * ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) - ngs.div(u) * q - ngs.div(v) * p - 1e-10 * p * q) * ngs.dx

   L_st = ngs.LinearForm(X)
   L_st += v * f * ngs.dx
   L_st += -v * h_outlet * ngs.ds(definedon=mesh.Boundaries('outlet'))

   c_st = ngs.Preconditioner(type="direct", bf=a_st, flags = {"inverse" : "umfpack" })
   a_st.Assemble()
   L_st.Assemble()

   gfu.components[0].Set(u_inlet, definedon=mesh.Boundaries('inlet'))
   ngs.BVP(bf=a_st, lf=L_st, gf=gfu, pre=c_st, maxsteps=100, prec=1e-10).Do()

   gfu_0.vec.data = gfu.vec

   # Construct the INS bilinear and linear forms. 
   a = ngs.BilinearForm(X)
   a += (u * v / dt_param) * ngs.dx
   a += (kv * ngs.InnerProduct(ngs.grad(u), ngs.grad(v)) - ngs.div(u) * q - ngs.div(v) * p - 1e-10 * p * q) * ngs.dx

   L = ngs.LinearForm(X)
   L += (gfu_0.components[0] * v / dt_param) * ngs.dx
   L += (v * f - ngs.InnerProduct(ngs.grad(gfu_0.components[0]) * gfu_0.components[0], v)) * ngs.dx
   L += v * h_outlet * ngs.ds(definedon=mesh.Boundaries('outlet'))

   # Construct the INS preconditioner.
   c = ngs.Preconditioner(type="direct", bf=a, flags = {"inverse" : "umfpack" })

   # Iterate through time.
   while t_param.Get() < t_end:
       t_param.Set(t_param.Get() + dt_param.Get())
    
       # Assemble the bilinear and linear forms and update the preconditioner.
       a.Assemble()
       L.Assemble()
       c.Update()

       # Apply the Dirichlet boundary conditions.
       gfu.components[0].Set(u_inlet, definedon=mesh.Boundaries('inlet'))

       # Single solve.
       ngs.BVP(bf=a, lf=L, gf=gfu, pre=c, maxsteps=20, prec=1e-10).Do()

       # Update the value of the previous time step.
       gfu_0.vec.data = gfu.vec
