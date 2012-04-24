====================================
Downloading and Installation
====================================

Prerequisites
~~~~~~~~~~~~~~~

The lmfit package requires Python, Numpy, and Scipy.  Extensive testing on
version compatibility has not yet been done.  Initial tests work with
Python 3.2, but no testing with Python 3 has yet been done.  No testing has
been done with 64-bit architectures, but as this package is pure Python, no
significant troubles are expected.


Downloads
~~~~~~~~~~~~~

The latest stable version is available from PyPI or CARS (Univ of Chicago):

.. _lmfit-0.5.tar.gz (CARS):   http://cars9.uchicago.edu/software/python/lmfit/src/lmfit-0.5.tar.gz
.. _lmfit-0.5.win32-py2.6.exe (CARS): http://cars9.uchicago.edu/software/python/lmfit/src/lmfit-0.5.win32-py2.6.exe
.. _lmfit-0.5.win32-py2.7.exe (CARS): http://cars9.uchicago.edu/software/python/lmfit/src/lmfit-0.5.win32-py2.7.exe
.. _lmfit-0.5.win32-py3.2.exe (CARS): http://cars9.uchicago.edu/software/python/lmfit/src/lmfit-0.5.win32-py3.2.exe

.. _lmfit-0.5.tar.gz (PyPI): http://pypi.python.org/packages/source/l/lmfit/lmfit-0.5.tar.gz
.. _lmfit-0.5.win32-py2.6.exe (PyPI): http://pypi.python.org/packages/any/l/lmfit/lmfit-0.5.win32-py2.6.exe
.. _lmfit-0.5.win32-py2.7.exe (PyPI): http://pypi.python.org/packages/any/l/lmfit/lmfit-0.5.win32-py2.7.exe
.. _lmfit-0.5.win32-py3.2.exe (PyPI): http://pypi.python.org/packages/any/l/lmfit/lmfit-0.5.win32-py3.2.exe

.. _lmfit github repository:   http://github.com/newville/lmfit-py
.. _lmfit at pypi:             http://pypi.python.org/pypi/lmfit/
.. _Python Setup Tools:        http://pypi.python.org/pypi/setuptools

+----------------------+------------------+--------------------------------------------+
|  Download Option     | Python Versions  |  Location                                  |
+======================+==================+============================================+
|  Source Kit          | 2.6, 2.7, 3.2    | -  `lmfit-0.5.tar.gz (PyPI)`_              |
|                      |                  | -  `lmfit-0.5.tar.gz (CARS)`_              |
+----------------------+------------------+--------------------------------------------+
|  Win32 Installer     |   2.6            | -  `lmfit-0.5.win32-py2.6.exe (PyPI)`_     |
|                      |                  | -  `lmfit-0.5.win32-py2.6.exe (CARS)`_     |
+----------------------+------------------+--------------------------------------------+
|  Win32 Installer     |   2.7            | -  `lmfit-0.5.win32-py2.7.exe (PyPI)`_     |
|                      |                  | -  `lmfit-0.5.win32-py2.7.exe (CARS)`_     |
+----------------------+------------------+--------------------------------------------+
|  Win32 Installer     |   3.2            | -  `lmfit-0.5.win32-py3.2.exe (PyPI)`_     |
|                      |                  | -  `lmfit-0.5.win32-py3.2.exe (CARS)`_     |
+----------------------+------------------+--------------------------------------------+
|  Development Version |   all            |  use `lmfit github repository`_            |
+----------------------+------------------+--------------------------------------------+

if you have `Python Setup Tools`_  installed, you can download and install
the lmfit-py Package simply with::

   easy_install -U lmfit


Development Version
~~~~~~~~~~~~~~~~~~~~~~~~

To get the latest development version, use::

   git clone http://github.com/newville/lmfit-py.git


Installation
~~~~~~~~~~~~~~~~~

Installation from source on any platform is::

   python setup.py install

Acknowledgements
~~~~~~~~~~~~~~~~~~

LMFIT was originally written by Matthew Newville.  Substantial code and
documentation improvements, especially for improved estimates of confidence
intervals was provided by Till Stensitzki.  Many valuable suggestions for
improvements have come from Christoph Deil.  The code obviously depends on,
and owes a very large debt to the code in scipy.optimize.  Several
discussions on the scipy mailing lists have also led to improvements in
this code.


License
~~~~~~~~~~~~~

The LMFIT-py code is distribution under the following license:

  Copyright (c) 2012 Matthew Newville, The University of Chicago
                     Till Stensitzki, Freie Universitat Berlin'

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


