"""
Benchmarks of methods with and without computing the Jacobian analytically
==========================================================================

Providing a function that calculates the Jacobian matrix analytically can
reduce the time spent finding a solution. The results from benchmarks comparing
two methods (``leastsq`` and ``least_squares``) with and without a function to
calculate the Jacobian matrix analytically are presented below.

First we define the model function, the residual function, and the appropriate
Jacobian functions:
"""
from timeit import timeit
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from lmfit import Parameters, minimize

NUM_JACOBIAN_CALLS = 0


def func(var, x):
    return var[0] * np.exp(-var[1]*x) + var[2]


def residual(pars, x, data):
    a, b, c = pars['a'], pars['b'], pars['c']
    model = func((a, b, c), x)
    return model - data


def dfunc(pars, x, data):
    global NUM_JACOBIAN_CALLS
    NUM_JACOBIAN_CALLS += 1

    a, b = pars['a'], pars['b']
    v = np.exp(-b*x)
    return np.array([v, -a*x*v, np.ones(len(x))])


def jacfunc(pars, x, data):
    global NUM_JACOBIAN_CALLS
    NUM_JACOBIAN_CALLS += 1

    a, b = pars['a'], pars['b']
    v = np.exp(-b*x)
    jac = np.ones((len(x), 3), dtype=np.float64)
    jac[:, 0] = v
    jac[:, 1] = -a * x * v
    return jac


a, b, c = 2.5, 1.3, 0.8

x = np.linspace(0, 4, 50)
y = func([a, b, c], x)

data = y + 0.15*np.random.RandomState(seed=2021).normal(size=x.size)


###############################################################################
# Then we define the different cases to benchmark (i.e., different methods with
# and without a function to calculate the Jacobian analytically) and the number
# of repetitions per case:
cases = (
    dict(
        method='leastsq',
    ),
    dict(
        method='leastsq',
        Dfun=dfunc,
        col_deriv=1,
    ),
    dict(
        method='least_squares',
    ),
    dict(
        method='least_squares',
        jac=jacfunc,
    ),
)

num_repeats = 100
results = []

for kwargs in cases:
    params = Parameters()
    params.add('a', value=10)
    params.add('b', value=10)
    params.add('c', value=10)

    wrapper = lambda: minimize(
        residual,
        params,
        args=(x,),
        kws={'data': data},
        **kwargs,
    )
    time = timeit(wrapper, number=num_repeats) / num_repeats

    NUM_JACOBIAN_CALLS = 0
    fit = wrapper()

    results.append(SimpleNamespace(
        time=time,
        num_jacobian_calls=NUM_JACOBIAN_CALLS,
        fit=fit,
        kwargs=kwargs,
    ))


###############################################################################
# Finally, we present the results:
labels = []

for result in results:
    label = result.kwargs['method']
    if result.num_jacobian_calls > 0:
        label += ' with Jac.'

    labels.append(label)

label_width = max(map(len, labels))
lines = [
    '| '
    + ' | '.join([
        'Method'.ljust(label_width),
        'Avg. time (ms)',
        '# func. (+ Jac.) calls',
        'Chi-squared',
        'a'.ljust(5),
        'b'.ljust(5),
        'c'.ljust(6),
    ])
    + '|'
]

print(f'The "true" parameters are: a = {a:.3f}, b = {b:.3f}, c = {c:.3f}\n')
fig, ax = plt.subplots()
ax.plot(x, data, marker='.', linestyle='none', label='data')

for (result, label) in zip(results, labels):
    linestyle = '-'
    if result.num_jacobian_calls > 0:
        linestyle = '--'

    a = result.fit.params['a'].value
    b = result.fit.params['b'].value
    c = result.fit.params['c'].value
    y = func([a, b, c], x)
    ax.plot(x, y, label=label, alpha=0.5, linestyle=linestyle)

    columns = [
        label.ljust(label_width),
        f'{result.time * 1000:.2f}'.ljust(14),
        (
            f'{result.fit.nfev}'
            + (
                f' (+{result.num_jacobian_calls})'
                if result.num_jacobian_calls > 0 else
                ''
            )
        ).ljust(22),
        f'{result.fit.chisqr:.3f}'.ljust(11),
        f'{a:.3f}'.ljust(5, '0'),
        f'{b:.3f}'.ljust(5, '0'),
        f'{c:.3f}'.ljust(5, '0'),
    ]
    lines.append('| ' + ' | '.join(columns) + ' |')

lines.insert(1, '|-' + '-|-'.join('-' * len(col) for col in columns) + '-|')
print('\n'.join(lines))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()
