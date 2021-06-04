Using Sphinx
============

**Getting Started**

Official guide to Sphinx can be found [here](https://www.sphinx-doc.org/en/master/usage/quickstart.html).

Another helpful guide is [this one](https://samnicholls.net/2016/06/15/how-to-sphinx-readthedocs/).



If you have cloned the OpenCMP repository you should have all the necessary configuration files setup. Try the following steps. 

1. Install Sphinx with `apt-get install python3-sphinx`. Make sure you install for Python 3.6+.
2. Check if you have docs/\_static and docs/\_templates directories. If you don't, create empty directories with these names.
3. Within the docs directory in the main OpenCMP directory run `make html`.
4. Move to the docs/\_build/html directory and open any of the .html files (index.html is a good one to start with).

If the above steps didn't work run `sphinx-quickstart` as detailed in the [official guide](https://www.sphinx-doc.org/en/master/usage/quickstart.html) then modify the conf.py file as described in the next section.



**Configuring the conf.py File**

I have the following lines uncommented/added at the top of the file.

```python
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))
```



I added `sphinx.ext.napoleon` to the end of `extensions` so Sphinx can parse Google-style docstrings. I also added the following lines directly below `extensions`.

```python
napoleon_google_docstring = True
napoleon_include_private_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
```



To make Sphinx play nicely with typing add `sphinx_autodoc_typehints` to `extensions`. This must be after `sphinx.ext.napoleon` or it won't work.



I set the theme to the ReadTheDocs theme (`html_theme = 'sphinx_rtd_theme'`) which required installing the theme with `pip3 install sphinx_rtd_theme`.



I'm using recommonmark to parse Markdown into reStructuredText. I installed it with `pip3 install --upgrade recommonmark` then added `recommonmark` to `extensions`. The `source_suffix` list/dictionary also needs to include `.md`.



I'm using aafigure for ASCII art. I installed it with `pip3 install aafigure` then added `aafigure.sphinxext` to `extensions`.



**Helpful Tips**

* When using Markdown files the title must be denoted by underlining it with "=" (ex: look at this file in plain text). Sphinx won't recognise "#" and you will get an error about a missing title.

* To specify that a function returns multiple values do the following within the docstring:

  ![](../_static/docstring_multiple_returns.png)

* To force line breaks or other formatting within the main docstring block do the following:

  ![](../_static/docstring_main_block_formatting.png)

* If Sphinx is displaying old docstrings or failing to find modules that have been renamed or removed delete the "source" folder and then run `sphinx-apidoc -o source/ ../` from within the "docs" folder to recreate the automatic docstring documentation.

* Running `make clean` will clear out anything cached in the "_build" directory.

* Sphinx automatically wraps math blocks in the LaTeX "split" environment, so & can be used to align multi-line equations.