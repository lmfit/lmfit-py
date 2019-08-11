from lmfit import Parameters
params = Parameters()
params.add('a',value = 0.1)
params.add('b',value = 0.2)
params.add('c',value = 0.3)
def output(params, header = ['parameter','value']):
    """
    orginizes parameters in Class Parameters into a nice 2D numpy asarray
    """
    from numpy import asarray, stack, vstack
    header = asarray(['name', 'value', 'init_value', 'stderr', 'max', 'min', 'vary'])
    parameter = asarray([par.name for name, par in params.items()])
    value = asarray([par.value for name, par in params.items()])
    init_value = asarray([par.init_value for name, par in params.items()])
    stderr = asarray([par.stderr for name, par in params.items()])
    max = asarray([par.max for name, par in params.items()])
    min = asarray([par.min for name, par in params.items()])
    vary = asarray([par.vary for name, par in params.items()])
    return vstack((header,stack((parameter,value,init_value,stderr,max,min,vary), axis =-1)))

def testfunction(params):
    getattr(self,'params')
