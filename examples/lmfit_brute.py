#!/usr/bin/env python
#<examples/lmfit_brute.py>

from __future__ import print_function
import copy

import numpy as np

from lmfit import minimize, Minimizer, Parameters

# EXAMPLE 1 #
# taken from: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html

# create a set of Parameters
params = Parameters()
params.add_many(
        ('a', 2, False, None, None, None),
        ('b', 3, False, None, None, None),
        ('c', 7, False, None, None, None),
        ('d', 8, False, None, None, None),
        ('e', 9, False, None, None, None),
        ('f', 10, False, None, None, None),
        ('g', 44, False, None, None, None),
        ('h', -1, False, None, None, None),
        ('i', 2, False, None, None, None),
        ('j', 26, False, None, None, None),
        ('k', 1, False, None, None, None),
        ('l', -2, False, None, None, None),
        ('scale', 0.5, False, None, None, None),
        ('x', 0.0, True, -4.0, 4.0, None, 0.25),
        ('y', 0.0, True, -4.0, 4.0, None, 0.25),
    )

# define functions
def f1(p):
    par = p.valuesdict()
    return (par['a'] * par['x']**2 + par['b'] * par['x'] * par['y'] +
            par['c'] * par['y']**2 + par['d']*par['x'] + par['e']*par['y'] + par['f'])

def f2(p):
    par = p.valuesdict()
    return (-1.0*par['g']*np.exp(-((par['x']-par['h'])**2 +
            (par['y']-par['i'])**2) / par['scale']))

def f3(p):
    par = p.valuesdict()
    return (-1.0*par['j']*np.exp(-((par['x']-par['k'])**2 +
            (par['y']-par['l'])**2) / par['scale']))

# define objective function: returns the array to be minimized
def f(params):
    return f1(params) + f2(params) + f3(params)


# 1. show the effect of 'Ns': finite bounds for varying parameters ('x' and 'y'),
# and brute_step = 0.25
fitter = Minimizer(f, params)
result = fitter.minimize(method='brute', Ns=10, keep=5)
grid_x = np.unique([par.ravel() for par in result.brute_grid][0])
grid_y = np.unique([par.ravel() for par in result.brute_grid][1])

print("==========================================="
      "\nExample 1, taken from scipy.optimize.brute:\n"
      "===========================================\n\n"
      "Varying parameters with finite bounds and brute_step = 0.25:")
print("   {}\n   {}".format(params['x'], params['y']))
print("\nUsing the brute method with"
      "\n\tresult = fitter.minimize(method='brute', keep=5) "
      "\nwill generate a 2-dimensional grid, where 'x' and 'y' vary between "
      "\n'min' and 'max' (exclusive) spaced by 'brute_step': "
      "\n\nx: {}\ny: {}".format(grid_x, grid_y))
print("\nThe objective function is evaluated on this grid, and the raw output "
      "\nfrom scipy.optimize.brute is stored in brute_<parname> attributes.")
print("\nFor further use with lmfit, a number (determined by 'keep') of best scoring "
      "\nresults from the brute force method is converted to lmfit.Parameters "
      "\nobjects and stored in the .candidates attribute.")
print("\nThe optimal result from the brute force method is:")
print("result.candidates[0].score = {:.3f} (value of objective function)".\
      format(result.candidates[0].score))
print("result.candidates[0].params = ")
result.candidates[0].params.pretty_print()


## EXAMPLE 2: see examples/doc_basic.py

# create data to be fitted
x = np.linspace(0, 15, 301)
data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
        np.random.normal(size=len(x), scale=0.2) )

# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """ model decaying sine wave, subtract data"""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data

# create a set of Parameters
params = Parameters()
params.add('amp', value=7, min=2.5)
params.add('decay', value=0.05)
params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2)
params.add('omega', value=3, max=5)

# perform grid search with brute force method
# add brute_step for parameters that do not have both bounds specified
params['amp'].set(brute_step=0.25)
params['decay'].set(brute_step=0.005)
params['omega'].set(brute_step=0.25)

print("\n\n===================================================================="
      "\nExample 2: demonstrate how the grid points are determined and how to"
      "\nperform a leastsq minimization on the best candidates."
      "\n====================================================================\n")

print("Initial parameters:")
params.pretty_print()

fitter = Minimizer(fcn2min, params, fcn_args=(x, data))

print("\nUsing the brute method with"
      "\n\tresult_brute = fitter.minimize(method='brute', Ns=25, keep=25)"
      "\nwill generate a grid, where the parameter ranges depend on the "
      "\nsettings for value/min/max/brute_step as shown below.\n")

shift_grid = \
np.array([-1.57079633, -1.43989663, -1.30899694, -1.17809725, -1.04719755,
       -0.91629786, -0.78539816, -0.65449847, -0.52359878, -0.39269908,
       -0.26179939, -0.13089969,  0.        ,  0.13089969,  0.26179939,
        0.39269908,  0.52359878,  0.65449847,  0.78539816,  0.91629786,
        1.04719755,  1.17809725,  1.30899694,  1.43989663,  1.57079633])
amp_grid = \
np.array([ 2.5 ,  2.75,  3.  ,  3.25,  3.5 ,  3.75,  4.  ,  4.25,  4.5 ,
        4.75,  5.  ,  5.25,  5.5 ,  5.75,  6.  ,  6.25,  6.5 ,  6.75,
        7.  ,  7.25,  7.5 ,  7.75,  8.  ,  8.25,  8.5 ])
omega_grid = \
np.array([-1.25, -1.  , -0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75,
        1.  ,  1.25,  1.5 ,  1.75,  2.  ,  2.25,  2.5 ,  2.75,  3.  ,
        3.25,  3.5 ,  3.75,  4.  ,  4.25,  4.5 ,  4.75])
decay_grid = \
np.array([ -1.00000000e-02,  -5.00000000e-03,   5.20417043e-18,
         5.00000000e-03,   1.00000000e-02,   1.50000000e-02,
         2.00000000e-02,   2.50000000e-02,   3.00000000e-02,
         3.50000000e-02,   4.00000000e-02,   4.50000000e-02,
         5.00000000e-02,   5.50000000e-02,   6.00000000e-02,
         6.50000000e-02,   7.00000000e-02,   7.50000000e-02,
         8.00000000e-02,   8.50000000e-02,   9.00000000e-02,
         9.50000000e-02,   1.00000000e-01,   1.05000000e-01])

print("* amp, lower bound and brute_step: 25 (Ns) points between 'min' ({}) "
      "\nand 'min + Ns*brute_step' ({}, not inclusive)."
      "\namp_grid = {}\n".format(params['amp'].min, params['amp'].min +
                                 25*params['amp'].brute_step, amp_grid))
print("* decay, no bounds, but value and brute_step: 25 (Ns) points between "
      "\n'value - (Ns//2)*brute_step' ({:.2f}) and 'value + (Ns//2)*brute_step' ({:.2f}, not inclusive)."
      "\ndecay_grid = {}\n".format(params['decay'].value - (25//2) *
                                   params['decay'].brute_step,
                                   params['decay'].value + (25//2) *
                                   params['decay'].brute_step, decay_grid))
print("* shift, finite bounds and no brute_step: 25 (Ns) points between 'min' "
      "and 'max'.\nshift_grid = {}\n".format(shift_grid))
print("* omega, upper bound and brute_step: 25 (Ns) points between 'max - "
      "Ns*brute_step' ({}) \nand 'max' ({}, not inclusive)."
      "\namp_grid = {}\n".format(params['amp'].min, params['amp'].min +
                                 25*params['amp'].brute_step, amp_grid))

result_brute = fitter.minimize(method='brute', Ns=25, keep=25)

print("Since we used keep=25, the twenty-five best scoring candidates from "
      "the brute force \nmethod are converted to lmfit.Parameters objects "
      "and stored in the .candidates attribute. \nNow we can perform a leastsq "
      "minimization on these candidates as follows:\n")
print("\tbest_result = copy.deepcopy(result_brute)"
      "\n\tfor candidate in result_brute.candidates:"
      "\n\t    trial = fitter.minimize(method='leastsq', params=candidate.params)"
      "\n\t    if trial.chisqr < best_result.chisqr:"
      "\n\t        best_result = trial")

best_result = copy.deepcopy(result_brute)
for candidate in result_brute.candidates:
    trial = fitter.minimize(method='leastsq', params=candidate.params)
    if trial.chisqr < best_result.chisqr:
        best_result = trial

print("\nBest candidate from brute force method: chi-sqr = {:.3f}".\
      format(result_brute.chisqr))
result_brute.params.pretty_print()

print("\nBest result after the leastsq minimization: chi-sqr = {:.3f}".\
      format(best_result.chisqr))
best_result.params.pretty_print()


def plot_results_brute(result, best_vals=True, varlabels=None,
                       output='results_brute'):
    """Visualize the result of the brute force grid search.

    The output file will display the chi-square value per parameter and contour
    plots for all combination of two parameters.

    Inspired by the `corner` package (https://github.com/dfm/corner.py).

    Parameters
    ----------
    result : :class:`~lmfit.minimizer.MinimizerResult`
        Contains the results from the :meth:`brute` method.

    best_vals : bool, optional
        Whether to show the best values from the grid search (default is True).

    varlabels : list, optional
        If None (default), use `result.var_names` as axis labels, otherwise
        use the names specified in `varlabels`.

    output : str, optional
        Name of the output PDF file (default is 'result_brute')
    """

    npars = len(result.var_names)
    fig, axes = plt.subplots(npars, npars)

    if not varlabels:
        varlabels = result.var_names
    if best_vals and isinstance(best_vals, bool):
        best_vals = result.params

    for i, par1 in enumerate(result.var_names):
        for j, par2 in enumerate(result.var_names):

            # parameter vs chi2 in case of only one parameter
            if npars == 1:
                axes.plot(result.brute_grid, result.brute_Jout, 'o', ms=3)
                axes.set_ylabel(r'$\chi^{2}$')
                axes.set_xlabel(varlabels[i])
                if best_vals:
                    axes.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on top
            elif i == j and j < npars-1:
                if i == 0:
                    axes[0, 0].axis('off')
                ax = axes[i, j+1]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.unique(result.brute_grid[i]),
                        np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        'o', ms=3)
                ax.set_ylabel(r'$\chi^{2}$')
                ax.yaxis.set_label_position("right")
                ax.yaxis.set_ticks_position('right')
                ax.set_xticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')

            # parameter vs chi2 profile on the left
            elif j == 0 and i > 0:
                ax = axes[i, j]
                red_axis = tuple([a for a in range(npars) if a != i])
                ax.plot(np.minimum.reduce(result.brute_Jout, axis=red_axis),
                        np.unique(result.brute_grid[i]), 'o', ms=3)
                ax.invert_xaxis()
                ax.set_ylabel(varlabels[i])
                if i != npars-1:
                    ax.set_xticks([])
                elif i == npars-1:
                    ax.set_xlabel(r'$\chi^{2}$')
                if best_vals:
                    ax.axhline(best_vals[par1].value, ls='dashed', color='r')

            # contour plots for all combinations of two parameters
            elif j > i:
                ax = axes[j, i+1]
                red_axis = tuple([a for a in range(npars) if a != i and a != j])
                X, Y = np.meshgrid(np.unique(result.brute_grid[i]),
                                   np.unique(result.brute_grid[j]))
                lvls1 = np.linspace(result.brute_Jout.min(),
                                    np.median(result.brute_Jout)/2.0, 7, dtype='int')
                lvls2 = np.linspace(np.median(result.brute_Jout)/2.0,
                                    np.median(result.brute_Jout), 3, dtype='int')
                lvls = np.unique(np.concatenate((lvls1, lvls2)))
                ax.contourf(X.T, Y.T, np.minimum.reduce(result.brute_Jout, axis=red_axis),
                            lvls, norm=LogNorm())
                ax.set_yticks([])
                if best_vals:
                    ax.axvline(best_vals[par1].value, ls='dashed', color='r')
                    ax.axhline(best_vals[par2].value, ls='dashed', color='r')
                    ax.plot(best_vals[par1].value, best_vals[par2].value, 'rs', ms=3)
                if j != npars-1:
                    ax.set_xticks([])
                elif j == npars-1:
                    ax.set_xlabel(varlabels[i])
                if j - i >= 2:
                    axes[i, j].axis('off')

    plt.savefig('{}.pdf'.format(output.split('.')[0]))


try:
    from matplotlib.colors import LogNorm
    import matplotlib.pyplot as plt
    import pylab
    pylab.plot(x, data, 'k+')
    pylab.plot(x, data + fcn2min(result_brute.params, x, data), 'b',
               label='brute: best candidate #1')
    pylab.plot(x, data + fcn2min(result_brute.candidates[24].params, x, data),
               'g', label='brute: candidate #25')
    pylab.plot(x, data + best_result.residual, 'r--',
               label='optimal leastsq result')
    pylab.legend(loc='best')

    plot_results_brute(result_brute)
except:
    pass


#<examples/lmfit_brute.py>
