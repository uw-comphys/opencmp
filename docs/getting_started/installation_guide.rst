.. Explains how to install OpenCMP.
.. _installation_guide:

Installation Guide
==================

To begin using OpenCMP:

1) Clone the `GitHub repository <https://github.com/uw-comphys/opencmp>`_.
2) Install the following dependencies:

* Python 3.7+ 
* NGSolve version 6.2.2101 or later (`link <https://ngsolve.org/downloads>`_)
* configparser (:code:`pip install configparser`)
* pyparsing (:code:`pip install pyparsing`)
* edt (:code:`pip install edt`)
* tabulate (:code:`pip install tabulate`)
* meshio (:code:`pip install meshio`)
* pytest (:code:`pip install pytest`)

3) Install the :code:`opencmp` module using :code:`pip3 install .` from the top-level directory. 
4) Optionally, run the unit tests :code:`python -m pytests` from the top-level directory.
5) Go through the tutorials and other examples found in the "examples/" folder referring to the `tutorial instructions <https://opencmp.io/tutorials/index.html>`_.

Dependencies
------------


.. warning:: OpenCMP is primarily tested on Linux (Ubuntu 20.04-21.01) and MacOS. The main functionality should work on Windows but users may have issues exporting results to .vtk.
