.. Explains how to install OpenCMP.
.. _installation_guide:

Installation Guide
==================

To begin using OpenCMP clone the `GitHub repository <https://github.com/uw-comphys/opencmp>`_ and install the dependencies listed below. It is recommended that new users start with the tutorials. Examples of the different models available in OpenCMP can also be found in the "Examples" folder.

Dependencies
------------

OpenCMP uses Python 3.7+ and requires the following libraries:

* NGSolve version 6.2.2101 or later
* edt (:code:`pip install edt`)
* meshio (:code:`pip install meshio`)
* numpy (:code:`pip install numpy`)
* pyparsing (:code:`pip install pyparsing`)
* pytest >= 6.2 (:code:`pip install pytest`)
* scipy (:code:`pip install scipy`)
* tabulate (:code:`pip install tabulate`)

.. warning:: OpenCMP is primarily tested on Linux (Ubuntu 20.04-21.01). The main functionality should work on Windows but users may have issues exporting results to .vtk.
