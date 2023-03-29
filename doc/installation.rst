============================
Downloading and Installation
============================

.. _lmfit github repository: https://github.com/lmfit/lmfit-py
.. _python: https://python.org
.. _scipy: https://scipy.org/scipylib/index.html
.. _numpy: https://numpy.org/
.. _pytest: https://pytest.org/
.. _pytest-cov: https://github.com/pytest-dev/pytest-cov
.. _emcee: https://emcee.readthedocs.io/
.. _pandas: https://pandas.pydata.org/
.. _jupyter: https://jupyter.org/
.. _matplotlib: https://matplotlib.org/
.. _dill: https://github.com/uqfoundation/dill
.. _asteval: https://github.com/newville/asteval
.. _uncertainties: https://github.com/lebigot/uncertainties
.. _numdifftools: https://github.com/pbrod/numdifftools
.. _contributing.md: https://github.com/lmfit/lmfit-py/blob/master/.github/CONTRIBUTING.md
.. _corner: https://github.com/dfm/corner.py
.. _sphinx: https://www.sphinx-doc.org
.. _jupyter_sphinx: https://jupyter-sphinx.readthedocs.io
.. _ipykernel: https://github.com/ipython/ipykernel
.. _sphinxcontrib-svg2pdfconverter: https://github.com/missinglinkelectronics/sphinxcontrib-svg2pdfconverter
.. _cairosvg: https://cairosvg.org/
.. _Pillow: https://python-pillow.org/
.. _sphinx-gallery: https://sphinx-gallery.github.io/stable/index.html
.. _flaky: https://github.com/box/flaky
.. _SymPy: https://www.sympy.org/
.. _Latexmk: https://ctan.org/pkg/latexmk/

Prerequisites
~~~~~~~~~~~~~

Lmfit works with `Python`_ versions 3.7 and higher. Version
0.9.15 is the final version to support Python 2.7.

Lmfit requires the following Python packages, with versions given:
   * `NumPy`_ version 1.19 or higher.
   * `SciPy`_ version 1.6 or higher.
   * `asteval`_ version 0.9.28 or higher.
   * `uncertainties`_ version 3.1.4 or higher.

All of these are readily available on PyPI, and are installed
automatically if installing with ``pip install lmfit``.

In order to run the test suite, the `pytest`_, `pytest-cov`_, and `flaky`_
packages are required. Some functionality requires the `emcee`_ (version 3+),
`corner`_, `pandas`_, `Jupyter`_, `matplotlib`_, `dill`_, or `numdifftools`_
packages. These are not installed automatically, but we highly recommend each
of them.

For building the documentation and generating the examples gallery, `matplotlib`_,
`emcee`_ (version 3+), `corner`_, `Sphinx`_, `sphinx-gallery`_, `jupyter_sphinx`_,
`ipykernel`_, `Pillow`_, and `SymPy`_ are required. For generating the PDF documentation,
the Python packages `sphinxcontrib-svg2pdfconverter`_ and `cairosvg`_ are also required,
as well as the LaTex package `Latexmk`_ (which is included by default in some
LaTex distributions).

Please refer to ``setup.cfg`` under ``options.extras_require`` for a list of all
dependencies that are needed if you want to participate in the development of lmfit.
You can install all these dependencies automatically by doing ``pip install lmfit[all]``,
or select only a subset (e.g., ``dev```, ``doc``, or ``test``).

Please note: the "original" ``python setup.py install`` is deprecated, but we will
provide a shim ``setup.py`` file for as long as ``Python`` and/or ``setuptools``
allow the use of this legacy command.

Downloads
~~~~~~~~~

The latest stable version of lmfit is |release| and is available from `PyPI
<https://pypi.python.org/pypi/lmfit/>`_. Check the :ref:`whatsnew_chapter` for
a list of changes compared to earlier releases.

Installation
~~~~~~~~~~~~

The easiest way to install lmfit is with::

    pip install lmfit

For Anaconda Python, lmfit is not an official package, but several
Anaconda channels provide it, allowing installation with (for example)::

   conda install -c conda-forge lmfit


Development Version
~~~~~~~~~~~~~~~~~~~

To get the latest development version from the `lmfit GitHub repository`_, use::

   git clone https://github.com/lmfit/lmfit-py.git

and install using::

   pip install --upgrade build pip setuptools wheel

to install the required build dependencies and then do::

   python -m build
   pip install ".[all]'

to generate the wheel and install ``lmfit`` with all its dependencies.

We welcome all contributions to lmfit! If you cloned the repository for this
purpose, please read `CONTRIBUTING.md`_ for more detailed instructions.

Testing
~~~~~~~

A battery of tests scripts that can be run with the `pytest`_ testing framework
is distributed with lmfit in the ``tests`` folder. These are automatically run
as part of the development process.
For any release or any master branch from the git repository, running ``pytest``
should run all of these tests to completion without errors or failures.

Many of the examples in this documentation are distributed with lmfit in the
``examples`` folder, and should also run for you. Some of these examples assume
that `matplotlib`_ has been installed and is working correctly.

Acknowledgements
~~~~~~~~~~~~~~~~

.. literalinclude:: ../AUTHORS.txt
  :language: none


Copyright, Licensing, and Re-distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LMFIT-py code is distributed under the following license:

.. literalinclude:: ../LICENSE
  :language: none
