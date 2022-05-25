.. Explains how to install OpenCMP.
.. _installation_guide:

Installation Guide
==================

.. warning:: OpenCMP is primarily tested on Linux (Ubuntu 20.04-21.01) and MacOS. The main functionality should work on Windows but users may have issues exporting results to .vtu.

.. warning:: NGSolve support for the Anaconda Python Distribution is currently experimental, please do not use NGSolve installed in this way with OpenCMP until further notice.

**NOTE** 

* Top-level directory refers to the highest level directory of OpenCMP. I.e., */users/.../opencmp*.

Installing OpenCMP
==================

1) Clone the `GitHub repository <https://github.com/uw-comphys/opencmp>`_.

2) Install :code:`pip3`:

    * Linux and WSL (Windows Subsystem for Linux) - Execute :code:`sudo apt install python3-pip`.
    
    * macOS - Install `Homebrew <https://brew.sh/>`_ via terminal. Then :code:`brew install python3`. This also installs pip3.
    
    * Windows - We recommend using `WSL <#supplementary-information>`_ (Windows Subsystem for Linux).

3) Install the :code:`opencmp` module: from the top-level directory, execute :code:`pip3 install .`

4) Optionally, run all the unit tests using :code:`pytesting` from the top-level directory. Note,

    * In order to run the tests, all optional dependencies must be installed.
    
    * For full information about running the tests see the README.md inside the pytests folder.

5) Go through the `tutorials <https://opencmp.io/tutorials/index.html>`_ and other examples found in the "examples/" folder.


Custom Commands for OpenCMP
===========================

**Installing Optional Dependencies**

* :code:`get_optional_dependencies` automatically installs all the optional dependencies.

* :code:`get_pytest` automatically installs the :code:`pytest` python library. 

**Running OpenCMP Test Files**

* :code:`pytesting` from the top-level directory runs all the pytests. 


Dependencies
============

**Required**

* Python 3.7+ 

**Optional**

* edt (:code:`pip3 install edt`)

    * Needed for the Diffuse Interface Method.

* tabulate (:code:`pip3 install tabulate`)

    * Needed to output results for mesh refinement and polynomial order convergence tests.

* pytest (:code:`pip3 install pytest`)

    * Needed for the unit tests.

Supplementary Information
=========================

**WSL (Windows Subsystem for Linux)**

* WSL (Windows Subsystem for Linux) is the recommended platform for OpenCMP for native Windows users.

* To install WSL, go to the `Microsoft Store <ms-windows-store://home>`_. Install Ubuntu 20.04 or 22.04.

* Python 3.7+ should come pre-installed. To check this, execute :code:`python3 --version`.

**Setting up WSL for OpenCMP**

* To be able to use the `custom commands for OpenCMP <#custom-commands-for-opencmp>`_, please perform the following. 

    * In WSL, execute :code:`nano ~/.bashrc`

    * At the bottom of the file append the line :code:`export PATH="/home/user/.local/bin:$PATH"` where :code:`user` is the username of your WSL unix profile.

    * Press :code:`CTRL+S` then :code:`CTRL+X`. Exit WSL and restart the application. 

* Ngsolve, a required dependency for OpenCMP, has a graphics issue for WSL. To correct this, execute :code:`sudo apt install ubuntu-desktop`

    * This installs a windows manager for WSL that will allow ngsolve to function correctly.

**Installing an X Server on WSL**

* WSL does not come with GUI (graphical user interface) application support (shortformed an "X server"). To view output plots or any graphical interface, perform the following.

    * Change display variables in WSL.
    
        * Execute :code:`nano ~/.bashrc`. At the bottom of the page, append :code:`export DISPLAY=$(ip route list default | awk '{print $3}'):0` and :code:`export LIBGL_ALWAYS_INDIRECT=1` (on separate lines).

        * Press :code:`CTRL+S` then :code:`CTRL+X`. Exit WSL and restart the application.
    
    * Enable Public Access on your X11 server for Windows. Follow the tutorial `here <https://skeptric.com/wsl2-xserver/>`_. Be sure to only follow the section "Allow WSL Access via Windows Firewall".

    * Download `VcXsrv https://sourceforge.net/projects/vcxsrv/>`_. Then:
    
        * Navigate to :code:`C:\Program Files\VcXsrv` and open :code:`xlaunch.exe`. Click Next until "Extra Settings" page.

        * Be sure to check the box for "Disable Access Control". Save the configuration file somewhere useful.

        * When you want to see visuals in WSL (i.e., :code:`matplotlib` or other outputs/plots), be sure to double click the config.xlaunch you just created before executing the code!






