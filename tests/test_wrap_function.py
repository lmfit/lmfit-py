
from lmfit import wrap_function

def test_wrap_function():
    get_names = lambda p: [p_key for p_key in p ]

    def func(A, b, c, d=5., e=10):
        return float(A + b + c + d)

    x0 = [1, 2, 3]
    para, f = wrap_function(func, x0)
    assert(get_names(para) == ['A', 'b', 'c', 'd', 'e'])
    y1 = f(para)
    y2 = func(*x0)
    assert(y1==y2)

    x0 = [1, 2, 3, 4]
    para, f = wrap_function(func, x0)
    assert(get_names(para) == ['A', 'b', 'c', 'd', 'e'])
    y1 = f(para)
    y2 = func(*x0)
    assert(y1==y2)

    x0 = [9.2, 2.0, 7.]
    para, f = wrap_function(func, x0)
    assert(get_names(para) == ['A', 'b', 'c', 'd', 'e'])
    y1 = f(para)
    y2 = func(*x0)
    assert(y1==y2)

test_wrap_function()
