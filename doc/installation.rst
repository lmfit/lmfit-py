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


Prerequisites
~~~~~~~~~~~~~~~

Lmfit works with `Python`_ versions 2.7 and 3.4 - 3.7. Support for Python 2.6
and 3.3 ended with lmfit versions 0.9.4 and 0.9.8, respectively.  Support
for 2.7 is expected to end in early 2019.

Lmfit requires the following Python packages, with versions given:
   * `six`_ version 1.10 or higher.
   * `NumPy`_ version 1.10 or higher.
   * `SciPy`_ version 0.19 or higher.
   * `asteval`_ version 0.9.12 or higher.
   * `uncertainties`_ version 3.0 or higher.

All of these are readily available on PyPI, and should be installed
automatically if installing with ``pip install lmfit``.

In order to run the test suite, the `pytest`_ package is required.  Some
functionality requires the `emcee`_, `pandas`_, `Jupyter`_, `matplotlib`_,
`dill`_, or `numdifftools`_ packages.  These are not installed
automatically, but we highly recommend each of these packages.


Downloads
~~~~~~~~~~~~~

The latest stable version of lmfit is |release| and is available from `PyPi
<https://pypi.python.org/pypi/lmfit/>`_.

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
