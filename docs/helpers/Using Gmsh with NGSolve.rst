.. Some tips for making Gmsh meshes that are compatible with NGSolve.
.. _gmsh_tips:

Using Gmsh with NGSolve
=======================

Reading in Gmsh Files
---------------------

Gmsh mesh files (.msh) can be read into Netgen using the :code:`netgen.read_gmsh` module. Note that the result will be a Netgen mesh, not an NGSolve mesh.

.. code-block:: python

   from netgen.read_gmsh import ReadGmsh
   ngmesh = ReadGmsh(mesh_filename)

Marking Boundaries
------------------

One of the benefits of meshing in Gmsh instead of Netgen is how much easier it is to mark different boundary locations. This is done by adding physical groups to the mesh.

.. note::
   If you are using physical groups to mark boundaries you must add them in the following order or Netgen will not be able to load them (ignore volumes if in 2D).

   1. Volumes
   2. Surfaces
   3. Lines
   4. Points

Once you have loaded your GMSH mesh into Netgen and converted it into an NGSolve mesh you can use :code:`mesh.GetBoundaries()` to see a list of the marked boundary sections. You may see the same name multiple times if that boundary section encompassed multiple surfaces in Gmsh. This will not affect the NGSolve solve as any boundary conditions assigned to that name will be applied on all mesh elements with that name.
