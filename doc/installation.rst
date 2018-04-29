====================================
Downloading and Installation
====================================

.. _lmfit github repository:   https://github.com/lmfit/lmfit-py
.. _python:  https://python.org
.. _scipy:  https://scipy.org/scipylib/index.html
.. _numpy:  http://numpy.org/
.. _nose:   https://nose.readthedocs.org/
.. _pytest: https://pytest.org/
.. _emcee:  http://dan.iel.fm/emcee/
.. _pandas:  https://pandas.pydata.org/
.. _jupyter:  https://jupyter.org/
.. _matplotlib:  https://matplotlib.org/
.. _dill:  https://github.com/uqfoundation/dill
.. _asteval:  https://github.com/newville/asteval
.. _six:  https://github.com/benjaminp/six
.. _uncertainties:  https://github.com/lebigot/uncertainties

Prerequisites
~~~~~~~~~~~~~~~

The lmfit package requires `Python`_, `NumPy`_, `SciPy`_, `asteval`_, `six`_,
and `uncertainties`_.

Lmfit works with Python versions 2.7, 3.4, 3.5, and 3.6. Support for Python 2.6
and 3.3 ended with lmfit versions 0.9.4 and 0.9.8, respectively. Scipy version
0.17, NumPy version 1.10, asteval version 0.9.12, six version 1.10, and
uncertainties version 3.0 are the minimum requirements.

In order to run the test suite, either the `nose`_ or `pytest`_ package is
required. Some functionality of lmfit requires the `emcee`_ package, some
functionality will make use of the `pandas`_, `Jupyter`_, `matplotlib`_,
or `dill`_ packages if available.  We highly recommend each of these
packages.


Downloads
~~~~~~~~~~~~~

The latest stable version of lmfit is |release| and is available from `PyPi
<https://pypi.python.org/pypi/lmfit/>`_.

Installation
~~~~~~~~~~~~~~~~~

With ``pip`` now widely avaliable, you can install lmfit with::

    pip install lmfit

Alternatively, you can download the source kit, unpack it and install with::

   python setup.py install

For Anaconda Python, lmfit is not an official package, but several
Anaconda channels provide it, allowing installation with (for example)::

   conda install -c GSECARS lmfit

or::

   conda install -c conda-forge lmfit


Development Version
~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version, use::

   git clone https://github.com/lmfit/lmfit-py.git

and install using::

   python setup.py install


Testing
~~~~~~~~~~

A battery of tests scripts that can be run with either the `nose`_ or
`pytest`_ testing framework is distributed with lmfit in the ``tests``
folder.  These are automatically run as part of the development process.
For any release or any master branch from the git repository, running
``pytest`` or ``nosetests`` should run all of these tests to completion
without errors or failures.

Many of the examples in this documentation are distributed with lmfit in
the ``examples`` folder, and should also run for you.  Some of these
examples assume `matplotlib`_ has been installed and is working correctly.

Acknowledgements
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../THANKS.txt


License
~~~~~~~~~~~~~

The LMFIT-py code is distribution under the following license:

.. literalinclude:: ../LICENSE
