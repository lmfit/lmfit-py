.. _bounds_chapter:

=====================
Bounds Implementation
=====================

.. _MINPACK-1: https://en.wikipedia.org/wiki/MINPACK
.. _MINUIT: https://en.wikipedia.org/wiki/MINUIT
.. _leastsqbound: https://github.com/jjhelmus/leastsqbound-scipy

This section describes the implementation of :class:`Parameter` bounds.
The `MINPACK-1`_ implementation used in :scipydoc:`optimize.leastsq` for
the Levenberg-Marquardt algorithm does not explicitly support bounds on
parameters, and expects to be able to fully explore the available range of
values for any Parameter. Simply placing hard constraints (that is,
resetting the value when it exceeds the desired bounds) prevents the
algorithm from determining the partial derivatives, and leads to unstable
results.

Instead of placing such hard constraints, bounded parameters are
mathematically transformed using the formulation devised (and documented)
for `MINUIT`_. This is implemented following (and borrowing heavily from)
the `leastsqbound`_ from J. J. Helmus. Parameter values are mapped from
internally used, freely variable values :math:`P_{\rm internal}` to bounded
parameters :math:`P_{\rm bounded}`. When both ``min`` and ``max`` bounds
are specified, the mapping is:

.. math::
   :nowrap:

   \begin{eqnarray*}
        P_{\rm internal} &=& \arcsin\big(\frac{2 (P_{\rm bounded} - {\rm min})}{({\rm max} - {\rm min})} - 1\big) \\
        P_{\rm bounded}  &=& {\rm min} + \big(\sin(P_{\rm internal}) + 1\big) \frac{({\rm max} - {\rm min})}{2}
    \end{eqnarray*}

With only an upper limit ``max`` supplied, but ``min`` left unbounded, the
mapping is:

.. math::
   :nowrap:

   \begin{eqnarray*}
        P_{\rm internal} &=& \sqrt{({\rm max} - P_{\rm bounded} + 1)^2 - 1} \\
        P_{\rm bounded}  &=& {\rm max} + 1 - \sqrt{P_{\rm internal}^2 + 1}
    \end{eqnarray*}

With only a lower limit ``min`` supplied, but ``max`` left unbounded, the
mapping is:

.. math::
   :nowrap:

   \begin{eqnarray*}
        P_{\rm internal} &=& \sqrt{(P_{\rm bounded} - {\rm min} + 1)^2 - 1} \\
        P_{\rm bounded}  &=& {\rm min} - 1 + \sqrt{P_{\rm internal}^2 + 1}
   \end{eqnarray*}

With these mappings, the value for the bounded Parameter cannot exceed the
specified bounds, though the internally varied value can be freely varied.

It bears repeating that code from `leastsqbound`_ was adopted to implement
the transformation described above. The challenging part (thanks again to
Jonathan J. Helmus!) here is to re-transform the covariance matrix so that
the uncertainties can be estimated for bounded Parameters. This is
included by using the derivate :math:`dP_{\rm internal}/dP_{\rm bounded}`
from the equations above to re-scale the Jacobin matrix before
constructing the covariance matrix from it. Tests show that this
re-scaling of the covariance matrix works quite well, and that
uncertainties estimated for bounded are quite reasonable. Of course, if
the best fit value is very close to a boundary, the derivative estimated
uncertainty and correlations for that parameter may not be reliable.

The `MINUIT`_ documentation recommends caution in using bounds. Setting
bounds can certainly increase the number of function evaluations (and so
computation time), and in some cases may cause some instabilities, as the
range of acceptable parameter values is not fully explored. On the other
hand, preliminary tests suggest that using ``max`` and ``min`` to set
clearly outlandish bounds does not greatly affect performance or results.
