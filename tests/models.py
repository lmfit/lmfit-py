# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 04:24:43 2012

@author: Tillsten
"""

from lmfit import Parameters, Minimizer
import numpy as np
import matplotlib.pyplot as plt

class LmModel(object):
    """ 
    Base class for all models. 
    
    Models take x and y and return 
    """
    def __init__(self,x,y):
        self.x, self.y=x,y
        self.parameters=Parameters()
        self.min=Minimizer(self.residual, self.parameters)
    
    def print_para(self):
        for i in self.parameters.values:
            print i
            
    def func(self,paras):
        raise NotImplementedError
    
    def est_startvals(self):
        raise NotImplementedError
    
    def residual(self,paras):
        return self.func(paras)-self.y
        
    def fit(self):
        self.min.leastsq()
        self.y_model=self.func(self.parameters)
        
    

class Linear(LmModel):
    """
    y = a*x + b
    """    
    def __init__(self,x,y):        
        LmModel.__init__(self,x,y)  
        self.parameters.add_many(('a',0), ('b',0))
        self.est_startvals()
        
    def est_startvals(self):        
        a, b = np.polyfit(self.x,self.y,1)
        self.parameters['a'].value = a
        self.parameters['b'].value = b 
               
    def func(self,paras):
        a=paras['a'].value 
        b=paras['b'].value 
        return a*self.x+b
        


class ExpDecay(LmModel):
    """
    y = a*exp(-x / b) + c
    """    
    def __init__(self,x,y):        
        LmModel.__init__(self,x,y)         
        self.parameters.add_many(('a',0), ('b',0),('c',0))
        self.est_startvals()
        
    def est_startvals(self):        
        c = np.min(self.y)
        a, b = np.polyfit(self.x, np.log(self.y-c+0.5),1)
        self.parameters['a'].value = np.exp(b)
        self.parameters['b'].value = 1/b
               
    def func(self,paras):
        a=paras['a'].value 
        b=paras['b'].value 
        c=paras['c'].value
        return a*np.exp(-x / b) + c
        
        
class Gaussian(LmModel):
    """
    y = a*exp(-(x-xc)**2/(2*w))+c
    """
    def __init__(self,x,y):        
        LmModel.__init__(self,x,y)         
        self.parameters.add_many(('a',0), ('xc',0),('w',0),('c',0))
        self.est_startvals()
        
    def est_startvals(self):        
        c = np.min(self.y)
        xc = x[np.argmax(abs(y))]
        a = np.max(y)
        w = abs(x[np.argmin(abs(a/2.-y))]-x[np.argmax(y)])*2
        self.parameters['c'].value=c
        self.parameters['xc'].value=xc
        self.parameters['a'].value=a
        self.parameters['w'].value=w
        
    def func(self,paras):
        c=paras['c'].value 
        xc=paras['xc'].value 
        a=paras['a'].value
        w=paras['w'].value
        return a*np.exp(-(self.x-xc)**2/(2*w))+c
    
        
#x=np.linspace(-5,5,20)
#y=3*x+1+np.random.randn(x.size)
#lm=Linear(x,y)
#lm.fit()

x=np.linspace(-5,5,20)
y= 5*np.exp(-x / 3.) + 3+ 4*np.random.randn(x.size)
lm=ExpDecay(x,y)
lm.fit()

plt.plot(lm.x, lm.y)
plt.plot(lm.x, lm.y_model)
plt.show()


        
        