.. Notes on how to get started with and use Sphinx.
.. _using_sphinx:

Using Sphinx
============

Getting Started
---------------

The official guide to Sphinx can be found `here <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_.
`This <https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/>`_ is another helpful guide.

If you have cloned the OpenCMP repository, you should have all the necessary Sphinx configuration files already set up. Try the following steps:

1. Install Sphinx with :code:`apt-get install python3-sphinx`. Make sure you install for Python 3.6+.
2. Check if you have "docs/_static" and "docs/_templates/" directories. If you don't, create empty directories with these names.
3. Within the "docs" directory in the main OpenCMP directory run :code:`make html`.
4. Move to the "docs/_build/html" directory and open any of the .html files ("index.html" is a good one to start with).

If any of the above steps don't work, run :code:`sphinx-quickstart` as detailed in the `official guide <https://www.sphinx-doc.org/en/master/usage/quickstart.html>`_ then modify the "conf.py" file as described in the next section.

Configuring "conf.py"
---------------------

The following lines must be uncommented or added to the top of the file.

.. code-block:: python

   import os
   import sys
   sys.path.insert(0, os.path.abspath('.'))
   sys.path.insert(0, os.path.abspath('../'))
   
Add :code:`sphinx.ext.napoleon` to the end of :code:`extensions` so Sphinx can parse Google-style docstrings. The following lines also need to be added directly below :code:`extensions`.

.. code-block:: python

   napoleon_google_docstring = True
   napoleon_include_private_with_doc = True
   napoleon_use_param = True
   napoleon_use_rtype = True
   
To make Sphinx play nicely with typing add :code:`sphinx_autodoc_typehints` to :code:`extensions`. This must be included after :code:`sphinx.ext.napoleon` or it won't work.

Set the Sphinx theme to the ReadTheDocs theme with :code:`html_theme = 'sphinx_rtd_theme'`. The theme can be installed with :code:`pip3 install sphinx_rdt_theme`.

Some OpenCMP documentation uses Markdown, which must be parsed into reStructuredText using recommonmark before it can be read by Sphinx. Install recommonmark with :code:`pip3 install --upgrade recommonmark` then add :code:`recommonmark` to :code:`extensions`. Finally, add :code:`.md` to the :code:`source_suffix` list/dictionary.

ASCII art is made with aafigure. Install aafigure with :code:`pip3 install aafigure` then add :code:`aafigure.sphinxext` to :code:`extensions`.

Helpful Tips
------------

* If using Markdown, the file titles must be denoted by underlining with "=". Sphinx doesn't recognize "#" and will give an error about a missing title.
* If Sphinx is displaying old docstrings or failing to find modules that have been renamed or removed, delete the "source" directory and then run :code:`sphinx-apidoc -o source/ ../` from within the "docs" directory. This will recreate the automatic docstring documentation from the current code base.
* Running :code:`make clean` clears anything cached in the "_build" directory.	
