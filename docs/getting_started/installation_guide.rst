.. Explains how to install OpenCMP.
.. _installation_guide:

Installation Guide
==================

.. warning:: OpenCMP is primarily tested on Linux (Ubuntu 20.04-21.01) and MacOS. The main functionality should work on Windows but users may have issues exporting results to .vtu.

.. warning:: NGSolve support for the Anaconda Python Distribution is currently experimental, please do not use NGSolve installed in this way with OpenCMP until further notice.

To begin using OpenCMP:

1) Clone the `GitHub repository <https://github.com/uw-comphys/opencmp>`_.
2) Install the dependencies
3) Install the :code:`opencmp` module using :code:`pip3 install .` from the top-level directory. 
4) Optionally, run the unit tests using :code:`python -m pytest` from the top-level directory. Note,

    * In order to run the tests, all optional dependencies must be installed.
    * For full information about running the tests see the README.md inside the pytests folder.

5) Go through the tutorials and other examples found in the "examples/" folder referring to the `tutorial instructions <https://opencmp.io/tutorials/index.html>`_.

Dependencies
------------
**Required**

* Python 3.7+
* NGSolve version 6.2.2105 or later (`link <https://ngsolve.org/downloads>`_)
* configparser (:code:`pip3 install configparser`)
* pyparsing (:code:`pip3 install pyparsing`)

**Optional**

* edt (:code:`pip3 install edt`)

    * Needed for the Diffuse Interface Method.

* tabulate (:code:`pip3 install tabulate`)

    * Needed to output results for mesh refinement and polynomial order convergence tests.

* pytest (:code:`pip3 install pytest`)

    * Needed for the unit tests.
