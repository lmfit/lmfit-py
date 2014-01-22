====================================
Downloading and Installation
====================================

Prerequisites
~~~~~~~~~~~~~~~

The lmfit package requires Python, Numpy, and Scipy.  Scipy version 0.11 or
higher is recommended, but extensive testing on version compatibility has
not been done.  Initial tests do work with Python 3.2, but little testing
with Python 3 has yet been done.  Scipy seems to not yet be available for
Python 3.3.  No testing has been done with 64-bit architectures, but as
this package is pure Python, no significant troubles are expected. Nose is
a requirement for running the test suite.

.. _uncertainties: http://packages.python.org/uncertainties/

If installed, the `uncertainties`_ package will be used for propagation of
uncertainties to constrained parameters.


Downloads
~~~~~~~~~~~~~

The latest stable version is available from PyPI:

.. _lmfit-0.7.4.tar.gz (PyPI): http://pypi.python.org/packages/source/l/lmfit/lmfit-0.7.4.tar.gz
.. _lmfit-0.7.4.win32-py2.6.exe (PyPI): http://pypi.python.org/packages/2.6/l/lmfit/lmfit-0.7.4.win32-py2.6.exe
.. _lmfit-0.7.4.win32-py2.7.exe (PyPI): http://pypi.python.org/packages/2.7/l/lmfit/lmfit-0.7.4.win32-py2.7.exe
.. _lmfit-0.7.4.win32-py3.2.exe (PyPI): http://pypi.python.org/packages/3.2/l/lmfit/lmfit-0.7.4.win32-py3.2.exe

.. _lmfit github repository:   http://github.com/lmfit/lmfit-py
.. _lmfit at pypi:             http://pypi.python.org/pypi/lmfit/
.. _Python Setup Tools:        http://pypi.python.org/pypi/setuptools

+----------------------+------------------+--------------------------------------------+
|  Download Option     | Python Versions  |  Location                                  |
+======================+==================+============================================+
|  Source Kit          | 2.6, 2.7, 3.2    | -  `lmfit-0.7.4.tar.gz (PyPI)`_            |
+----------------------+------------------+--------------------------------------------+
|  Win32 Installer     |   2.6            | -  `lmfit-0.7.4.win32-py2.6.exe (PyPI)`_   |
+----------------------+------------------+--------------------------------------------+
|  Win32 Installer     |   2.7            | -  `lmfit-0.7.4.win32-py2.7.exe (PyPI)`_   |
+----------------------+------------------+--------------------------------------------+
|  Win32 Installer     |   3.2            | -  `lmfit-0.7.4.win32-py3.2.exe (PyPI)`_   |
+----------------------+------------------+--------------------------------------------+
|  Development Version |   all            |  use `lmfit github repository`_            |
+----------------------+------------------+--------------------------------------------+

if you have `Python Setup Tools`_  installed, you can download and install
the lmfit-py Package simply with::

   easy_install -U lmfit


Development Version
~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version, use::

   git clone http://github.com/lmfit/lmfit-py.git


Installation
~~~~~~~~~~~~~~~~~

Installation from source on any platform is::

   python setup.py install

Acknowledgements
~~~~~~~~~~~~~~~~~~

LMFIT was originally written by Matthew Newville.  Substantial code and
documentation improvements, especially for improved estimates of confidence
intervals was provided by Till Stensitzki.  The implementation of parameter
bounds as described in the MINUIT documentation is taken from Jonathan
J. Helmus' leastsqbound code, with permission. Many valuable suggestions
for improvements have come from Christoph Deil.  The code obviously depends
on, and owes a very large debt to the code in scipy.optimize.  Several
discussions on the scipy mailing lists have also led to improvements in
this code.

License
~~~~~~~~~~~~~

The LMFIT-py code is distribution under the following license:

  Copyright (c) 2012 Matthew Newville, The University of Chicago
                     Till Stensitzki, Freie Universitat Berlin

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


