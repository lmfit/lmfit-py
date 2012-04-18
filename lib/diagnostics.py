# -*- coding: utf-8 -*-
"""
"""
from scipy.stats import f
from scipy.optimize import brentq


def calc_max_chi(N,P,best_chi):
    fval=f.isf(0.05,P,N-P)
    return best_chi*(fval*P/float(N-P)+1)

def f_compare(N,P,new_chi,best_chi):
    """Returns the probalitiy for two given parameter sets"""  
    #print new_chi, best_chi, N, P
    return f.cdf((new_chi/best_chi-1)*(N-P),1,N-P)


def copy_vals(params, tmp_params):
    "Saves the values of params in temporay dict"
    for para_key in params:
        tmp_params[para_key]=params[para_key].value

def restore_vals(tmp_params,params):
    "Restores the values of params from a temporay dict"
    for para_key in params:
        params[para_key].value=tmp_params[para_key]

def calc_ci(minimizer, maxiter=20, verbose=1):
    "Calculates coinfidance intervals using the model comparison method."
    fit_params=minimizer.params
    #copy the best fit values.
    org={}
    copy_vals(fit_params,org)
    output=[]
    best_chi=minimizer.chisqr
    
    for para in fit_params.values():        
        if verbose:
            print 'Calculating CI for '+ para.name
        restore_vals(org,fit_params)
        step=para.stderr
        para.vary=False    
        start_val=para.value
        #minimizer.leastsq()                
        
        def prob_func(val, offset=0.):
            "Returns the probabilty for given Value."
            para.value=val
            minimizer.prepare_fit(para)
            minimizer.leastsq()
            out=minimizer
            #print calc_max_chi(out.ndata, out.ndata-out.nfree,best_chi)
            prob=f_compare(out.ndata,out.ndata-out.nfree,out.chisqr,best_chi)                    
            return prob-offset
                        
        def search_limits(direction):
            """
            Searchs for the limits. First it looks for a upper limit and 
            then finds the sigma-limits with help of scipy root finder.
            """ 
            prob=0
            i=0
            limit=start_val
            #Find a upper limit,
            while prob<0.999:
                i+=1
                limit+=step*direction
                prob=prob_func(limit)
                if i>maxiter:
                    break
                
            restore_vals(org,fit_params)
            #use brentq to find sigmas.
            sigmas=[0.674,0.95,0.997]
            ret = [brentq(prob_func,start_val,limit, args=(p)) for p in sigmas]
            return ret
        
        
        upper_err=search_limits(1)
        restore_vals(org,fit_params)        
        lower_err=search_limits(-1)
        
        para.vary=True
          
        output.append([para.name]+list(lower_err[::-1])+[start_val]+list(upper_err))
    return output