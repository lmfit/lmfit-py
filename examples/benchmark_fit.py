#!/usr/bin/env python

import cProfile
import pstats
from subprocess import PIPE, Popen

import numpy as np

from lmfit import Parameters, minimize


def get_git_version():
    proc = Popen(['git', 'rev-parse', '--short', 'HEAD'], stdout=PIPE)
    return proc.communicate()[0].strip()


# define objective function: returns the array to be minimized
def fcn2min(params, x, data):
    """model decaying sine wave, subtract data"""
    amp = params['amp']
    shift = params['shift']
    omega = params['omega']
    decay = params['decay']
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)
    return model - data


def run_fit(nruns=100):
    # create data to be fitted
    x = np.linspace(0, 15, 601)
    np.random.seed(201)
    for i in range(nruns):
        data = (5. * np.sin(2 * x - 0.1) * np.exp(-x*x*0.025) +
                np.random.normal(size=len(x), scale=0.1))
        params = Parameters()
        params.add('amp', value=1.0, min=0.0, max=100.0)
        params.add('decay', value=0.0, min=-1.0, max=10.0)
        params.add('shift', value=0.0, min=-np.pi/2., max=np.pi/2)
        params.add('omega', value=1.0, min=-10.0, max=10.0)
        out = minimize(fcn2min, params, args=(x, data))
        # print(out.params['amp'])
        assert out.params['amp'].value < 5.5
        assert out.params['amp'].value > 4.5
        assert out.params['omega'].value < 2.25
        assert out.params['omega'].value > 1.75
        # print(out.params['amp'])


def show_profile(filename):
    stats = pstats.Stats(filename)
    stats.strip_dirs().sort_stats('tottime').print_stats(20)


def profile_command(command, filename=None):
    gitversion = get_git_version()
    if filename is None:
        filename = '%s.prof' % gitversion
    prof = cProfile.run(command, filename=filename)
    show_profile(filename)


profile_command('run_fit()')
