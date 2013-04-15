# -*- coding: utf-8 -*-

import lmfit


def test_wrap_function():
    get_names = lambda p: [p_key for p_key in p ]

    def func(A, b, c, d=5, e=10):
        return A + b + c + d

    x0 = [1, 2, 3]
    para, f = lmfit.make_paras_and_func(func, x0)
    assert(get_names(para) == ['A', 'b', 'c'])
    y1 = f(para)
    y2 = func(*x0)
    assert(y1==y2)

    x0 = [1, 2, 3, 4]
    para, f = lmfit.make_paras_and_func(func, x0)
    assert(get_names(para) == ['A', 'b', 'c', 'd'])
    y1 = f(para)
    y2 = func(*x0)
    assert(y1==y2)

    x0 = [1, 2, 3]
    para, f = lmfit.make_paras_and_func(func, x0, {'e': 3})
    assert(get_names(para) == ['A', 'b', 'c', 'e'])
    y1 = f(para)
    y2 = func(*x0)
    assert(y1==y2)
