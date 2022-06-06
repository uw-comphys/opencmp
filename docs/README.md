To build the documentation install,

> pip3 install sphinx sphinxcontrib-aafig sphinx-autodoc-typehints recommonmark sphinx-rtd-theme

then build using the Makefile,

> make html

If Sphinx is displaying old docstrings or failing to find renamed/removed modules,

> sphinx-apidoc -o source/ ../

then recompile.
