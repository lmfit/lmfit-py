====================================
Downloading and Installation
====================================

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py
.. _python:  http://python.org
.. _scipy:  http://scipy.org/scipylib/index.html
.. _numpy:  http://numpy.org/
.. _nose:   http://nose.readthedocs.org/
.. _pytest: http://pytest.org/
.. _emcee:  http://dan.iel.fm/emcee/
.. _pandas:  http://pandas.pydata.org/
.. _jupyter:  http://jupyter.org/
.. _matplotlib:  http://matplotlib.org/

Prerequisites
~~~~~~~~~~~~~~~

The lmfit package requires `Python`_, `NumPy`_, and `SciPy`_.

Lmfit works with Python versions 2.7, 3.3, 3.4, 3.5, and 3.6. Support for Python 2.6
ended with lmfit version 0.9.4.  Scipy version 0.15 or higher is required,
with 0.17 or higher recommended to be able to use the latest optimization
features.  NumPy version 1.5.1 or higher is required.

In order to run the test suite, either the `nose`_ or `pytest`_ package is
required.  Some functionality of lmfit requires the `emcee`_ package, some
functionality will make use of the `pandas`_, `Jupyter`_ or `matplotlib`_
packages if available.  We highly recommend each of these
packages.


Downloads
~~~~~~~~~~~~~

The latest stable version of lmfit is |release| is available from `PyPi
<http://pypi.python.org/pypi/lmfit/>`_.

Installation
~~~~~~~~~~~~~~~~~

With ``pip`` now widely avaliable, you can install lmfit with::

    pip install lmfit

Alternatively, you can download the source kit, unpack it and install with::

   python setup.py install

For Anaconda Python, lmfit is not an official package, but several
Anaconda channels provide it, allowing installation with (for example)::

   conda install -c conda-forge lmfit

Development Version
~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version, use::

   git clone http://github.com/lmfit/lmfit-py.git

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
