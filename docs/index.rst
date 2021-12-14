.. OpenCMP documentation master file, created by
   sphinx-quickstart on Fri Apr 16 18:31:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OpenCMP's documentation!
===================================

OpenCMP is a computational multiphysics software package based on the finite element method. It is primarily intended for physicochemical processes involving significant convective flow. OpenCMP uses the NGSolve finite element library for spatial discretization and provides a configuration file-based interface to pre-implemented models and time discretization schemes. It provides built-in post-processing and error analysis and also integrates with Netgen, Gmsh, and ParaView for meshing and visualization of simulation results.

OpenCMP development follows the principles of ease of use, performance, and extensibility. The configuration file-based user interface is intended to be concise, readable, and intuitive. Similarly, the code base is structured such that experienced users can add their own models with minimal modifications to existing code. Inclusion of the finite element method enables the use of high-order polynomial interpolants for increased simulation accuracy. OpenCMP also offers the discontinuous Galerkin method which is locally conservative and improves simulation stability for convection-dominated problems. Finally, OpenCMP implements the diffuse interface method, a form of immersed boundary method which allows the use of non-conforming structured meshes for even complex simulation domains to improve simulation stability and sometimes speed.

The OpenCMP source code and examples and tutorial files are available on GitHub `<https://github.com/uw-comphys/opencmp>`_ subject to the GNU LGPL v2.1 open-source license.

.. toctree::
   :maxdepth: 1
   
   getting_started/index
   tutorials/index
   contributing/index
   mathematical_notes/index
   helpers/index
   source/modules
