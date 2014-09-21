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
are expected.  The `nose`_ frameworkt is required for running the test
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
the ``examples`` folder, and sould also run for you.  Many of these require


Acknowledgements
~~~~~~~~~~~~~~~~~~

LMFIT was originally written by Matthew Newville.  Substantial code and
documentation improvements, especially for improved estimates of confidence
intervals was provided by Till Stensitzki.  Much of the work on improved
unit testing and high-level model functions was done by Daniel B. Allen,
with substantial input from Antonino Ingargiola.  Many valuable suggestions
for improvements have come from Christoph Deil.  The implementation of
parameter bounds as described in the MINUIT documentation is taken from
Jonathan J. Helmus' leastsqbound code, with permission.  The code for
propagation of uncertainties is taken from Eric O. Le Bigot's uncertainties
package, with permission.  The code obviously depends on, and owes a very
large debt to the code in scipy.optimize.  Several discussions on the scipy
mailing lists have also led to improvements in this code.

License
~~~~~~~~~~~~~

The LMFIT-py code is distribution under the following license:

  Copyright (c) 2014 Matthew Newville, The University of Chicago, Till
  Stensitzki, Freie Universitat Berlin, Daniel B. Allen, Johns Hopkins
  University, Antonino Ingargiola, University of California, Los Angeles

  Permission to use and redistribute the source code or binary forms of this
  software and its documentation, with or without modification is hereby
  granted provided that the above notice of copyright, these terms of use,
  and the disclaimer of warranty below appear in the source code and
  documentation, and that none of the names of above institutions or
  authors appear in advertising or endorsement of works derived from this
  software without specific prior written permission from all parties.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
  DEALINGS IN THIS SOFTWARE.


