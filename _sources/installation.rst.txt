====================================
Downloading and Installation
====================================

.. _lmfit github repository:   https://github.com/lmfit/lmfit-py
.. _python:  https://python.org
.. _scipy:  https://scipy.org/scipylib/index.html
.. _numpy:  http://numpy.org/
.. _pytest: https://pytest.org/
.. _emcee:  http://dan.iel.fm/emcee/
.. _pandas:  https://pandas.pydata.org/
.. _jupyter:  https://jupyter.org/
.. _matplotlib:  https://matplotlib.org/
.. _dill:  https://github.com/uqfoundation/dill
.. _asteval:  https://github.com/newville/asteval
.. _six:  https://github.com/benjaminp/six
.. _uncertainties:  https://github.com/lebigot/uncertainties
.. _numdifftools:  https://github.com/pbrod/numdifftools
.. _contributing.md: https://github.com/lmfit/lmfit-py/blob/master/.github/CONTRIBUTING.md
.. _corner:  https://github.com/dfm/corner.py
.. _sphinx: https://www.sphinx-doc.org
.. _jupyter_sphinx: https://jupyter-sphinx.readthedocs.io
.. _ImageMagick: https://www.imagemagick.org/
.. _release_notes: https://lmfit.github.io/lmfit-py/whatsnew.html


Prerequisites
~~~~~~~~~~~~~~~

Lmfit works with `Python`_ versions 2.7 and 3.5, 3.6, 3.7, or 3.8. Version
0.9.15 is the final version to support Python 2.7.

Lmfit requires the following Python packages, with versions given:
   * `six`_ version 1.11 or higher.
   * `NumPy`_ version 1.16 or higher.
   * `SciPy`_ version 1.2 or higher.
   * `asteval`_ version 0.9.16 or higher.
   * `uncertainties`_ version 3.0.1 or higher.

All of these are readily available on PyPI, and should be installed
automatically if installing with ``pip install lmfit``.

In order to run the test suite, the `pytest`_ package is required.  Some
functionality requires the `emcee`_ (version 3+), `corner`_, `pandas`_, `Jupyter`_,
`matplotlib`_, `dill`_, or `numdifftools`_ packages.  These are not installed
automatically, but we highly recommend each of these packages.

For building the documentation, `matplotlib`_, `emcee`_ (version 3+), `corner`_,
`Sphinx`_, `jupyter_sphinx`_, and `ImageMagick`_ are required (the latter
one only when generating the PDF document).

Please refer to  ``requirements-dev.txt`` for a list of all dependencies that
are needed if you want to participate in the development of lmfit.

Downloads
~~~~~~~~~~~~~

The latest stable version of lmfit is |release| and is available from `PyPI
<https://pypi.python.org/pypi/lmfit/>`_. Check the `release_notes`_ for a list
of changes compared to earlier releases.

Installation
~~~~~~~~~~~~~~~~~

The easiest way to install lmfit is with::

    pip install lmfit

For Anaconda Python, lmfit is not an official package, but several
Anaconda channels provide it, allowing installation with (for example)::

   conda install -c GSECARS lmfit

or::

   conda install -c conda-forge lmfit


Development Version
~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version from the `lmfit GitHub repository`_, use::

   git clone https://github.com/lmfit/lmfit-py.git

and install using::

   python setup.py install

We welcome all contributions to lmfit! If you cloned the repository for this
purpose, please read `CONTRIBUTING.md`_ for more detailed instructions.

Testing
~~~~~~~~~~

A battery of tests scripts that can be run with the `pytest`_ testing framework
is distributed with lmfit in the ``tests`` folder. These are automatically run
as part of the development process.
For any release or any master branch from the git repository, running ``pytest``
should run all of these tests to completion without errors or failures.

Many of the examples in this documentation are distributed with lmfit in the
``examples`` folder, and should also run for you. Some of these examples assume
that  `matplotlib`_ has been installed and is working correctly.

Acknowledgements
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../THANKS.txt
  :language: none


Copyright, Licensing, and Re-distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The LMFIT-py code is distributed under the following license:

.. literalinclude:: ../LICENSE
  :language: none
