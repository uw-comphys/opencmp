.. Guidelines for new contributors.
.. _contributor_guidelines:

Contributor Guidelines
======================

How to Contribute
-----------------

#. Fork the repository.
#. Make your desired changes.
#. Document code changes with docstrings.
#. Add automated tests.

   * If adding new models or new time discretization schemes please add appropriate integration tests to "pytests/full_system".
   * If adding new functions or classes please add unit tests to "pytests" within an appropriate subdirectory.
   * Please update any tests affected by your code changes.
   * If adding or modifying documentation please check that Sphinx is able to compile it correctly. Follow the directions in :ref:`using_sphinx` to make the .html pages, confirm there are no warnings or errors, and open the .html pages locally to check their appearance.

#. Confirm that all tests run by running :code:`pytests` from the main OpenCMP directory.
#. Submit a pull request describing your changes.

Contributing to the OpenCMP Code Base
-------------------------------------

The OpenCMP developers welcome contributions to the code base to fix bugs, optimize existing code, or add new functionality. 

If you wish to add new models or new time discretization schemes it may be helpful to first read through :ref:`adding_models` and :ref:`adding_time_schemes`. Experience with the NGSolve finite element library is also necessary. Their documentation and examples can be found `here <https://docu.ngsolve.org/latest/>`_. For a short example of a typical NGSolve simulation script see :ref:`example_code`.

Contributing to OpenCMP Documentation
-------------------------------------

The OpenCMP developers also welcome contributions to improve existing documentation or add new documentation.

OpenCMP documentation takes various forms: type hinting, docstrings to outline the specific functionality of each function and class, tutorials to introduce new users to OpenCMP's functionality in short examples, example configuration files for each model for users to copy and then customize to their specific application, and auxilliary documentation for more detailed information on certain components of OpenCMP. 

All OpenCMP code should contain type hinting for readability and to help catch bugs. Use of `Mypy <http://mypy-lang.org/>`_ is suggested.

OpenCMP uses Google-style docstrings. The general syntax should be fairly intuitive; just copy existing docstrings. :ref:`writing_documentation` has tips for formatting more elaborate docstrings.

If you add new models to OpenCMP example configuration files for the new models would be appreciated. These should include all new configuration file parameters added specifically for the new models. Please also update :ref:`example_config` and :ref:`syntax` as necessary. 

Additions to the tutorials or new tutorials are welcome. However, if adding new tutorials, please discuss with the OpenCMP developers where these tutorials should fit in the general tutorial order. Additions to auxilliary documentation should also be discussed with the OpenCMP developers first.

Automated Tests
---------------

OpenCMP uses `Pytest <https://docs.pytest.org/en/6.2.x/>`_ for automated testing. All test code should go in the "pytests" directory. Unit tests are organized by the module being tested; each module has its own test script and these test scripts are organized in subdirectories named after the package for the module being tested. Integration tests go in the "full_system" subdirectory organized by the model being tested. Integration tests should use problems with known analytical solutions so accuracy can be assessed. Each model should be tested with all relevant time discretization schemes. If certain tests are expected to fail (ex: certain time discretization schemes are known to have bugs) the tests should be commented out but not deleted from the integration test scripts. Pull requests will be refused if they break any previously working integration tests.
