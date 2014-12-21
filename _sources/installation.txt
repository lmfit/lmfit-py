====================================
Downloading and Installation
====================================

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py
.. _Python Setup Tools:        http://pypi.python.org/pypi/setuptools
.. _pip:  https://pip.pypa.io/
.. _nose: http://nose.readthedocs.org/

Prerequisites
~~~~~~~~~~~~~~~

The lmfit package requires Python, Numpy, and Scipy.  Scipy version 0.13 or
higher is recommended, but extensive testing on compatibility with various
versions of scipy has not been done.  Lmfit does work with Python 2.7, and
3.2 and 3.3.  No testing has been done with Python 3.4, but as the package
is pure Python, relying only on scipy and numpy, no significant troubles
are expected.  The `nose`_ framework is required for running the test
suite, and IPython and matplotib are recommended.  If Pandas is available,
it will be used in portions of lmfit.


Downloads
~~~~~~~~~~~~~


The latest stable version of lmfit is  available from `PyPi <http://pypi.python.org/pypi/lmfit/>`_.

Installation
~~~~~~~~~~~~~~~~~

If you have `pip`_  installed, you can install lmfit with::

    pip install lmfit

or, if  you have `Python Setup Tools`_  installed, you install lmfit with::

   easy_install -U lmfit


or, you can download the source kit, unpack it and install with::

   python setup.py install


Development Version
~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version, use::

   git clone http://github.com/lmfit/lmfit-py.git


and install using::

   python setup.py install


Testing
~~~~~~~~~~

A battery of tests scripts that can be run with the `nose`_ testing
framework is distributed with lmfit in the ``tests`` folder.  These are
routinely run on the development version.  Running ``nosetests`` should run
all of these tests to completion without errors or failures.

Many of the examples in this documentation are distributed with lmfit in
the ``examples`` folder, and should also run for you.  Many of these require


Acknowledgements
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../THANKS.txt


License
~~~~~~~~~~~~~

The LMFIT-py code is distribution under the following license:

.. literalinclude:: ../LICENSE
