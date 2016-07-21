====================================
Downloading and Installation
====================================

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py
.. _Python Setup Tools:        http://pypi.python.org/pypi/setuptools
.. _pip:  https://pip.pypa.io/
.. _nose: http://nose.readthedocs.org/

Prerequisites
~~~~~~~~~~~~~~~

The lmfit package requires Python, Numpy, and Scipy.

Lmfit works with Python 2.7, 3.3, 3.4, and 3.5. Lmfit version 0.9.4 works
with Python 2.6, but support for it will dropped in version 0.9.5.  Scipy
version 0.13 or higher is required, with 0.17 or higher recommended to be
able to use the latest optimization features from scipy.  Support for scipy
0.13 will be dropped in version 0.9.5.  Numpy version 1.5 or higher is
required.

In order to run the test suite, the `nose`_ framework is required.  Some
parts of lmfit will be able to make use of IPython (version 4 or higher),
matplotlib, and pandas if those libraries are installed, but no core
functionality of lmfit requires these.


Downloads
~~~~~~~~~~~~~

The latest stable version of lmfit is |release| is available from `PyPi
<http://pypi.python.org/pypi/lmfit/>`_.

Installation
~~~~~~~~~~~~~~~~~

If you have `pip`_ installed, you can install lmfit with::

    pip install lmfit

or you can download the source kit, unpack it and install with::

   python setup.py install

For Anaconda Python, lmfit is not an official packages, but several
Anaconda channels provide it, allowing installation with (for example)::

   conda install -c newville lmfit


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
