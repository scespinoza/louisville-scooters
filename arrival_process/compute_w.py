import geopandas as gpd
import pandas as pd
from itertools import product
from statsmodels.tsa.stattools import acf
from math import exp, sin, pi
import numpy as np
from pyOpt import Optimization
from pyOpt.pyKSOPT.pyKSOPT import KSOPT
from pyOpt.pyALHSO.pyALHSO import ALHSO
from pyOpt.pyALPSO.pyALPSO import ALPSO
from pyOpt.pyNSGA2.pyNSGA2 import NSGA2
from pyOpt.pySLSQP.pySLSQP import SLSQP
from pyOpt.pyALGENCAN.pyALGENCAN import ALGENCAN 
from pyOpt.pyMIDACO.pyMIDACO import MIDACO
from pyOpt.pyCOBYLA.pyCOBYLA import COBYLA
import sys
sys.path.append('./modules/')
from helper_funcs import save_pickle, load_pickle

def w(x, teta, T_1=24, T_2=168):
    rho_1 = teta[0]**(x)
    rho_2 = teta[1]**(x)
    rho_3 = teta[2]**(sin(pi*x/T_1)*sin(pi*x/T_1))
    rho_4 = teta[3]**(sin(pi*x/T_2)*sin(pi*x/T_2))
    return rho_1 + rho_2*rho_3*rho_4

if __name__ == '__main__':
    
    print 'Loading data..'
    dataname = '../shapes/arrivals/arrivals.shp'
    data = gpd.read_file(dataname)
    region = gpd.read_file('../shapes/grid/grid_stkde.shp')
    region.drop(['bottom', 'top', 'left', 'right'], axis=1, inplace=True)
    region = region.to_crs(data.crs)
    joined_data = gpd.sjoin(data, region, how='left')
    joined_data['date'] = pd.to_datetime(joined_data['date'])
    joined_data['hour'] = joined_data['date'].dt.hour
    joined_data['cum_hour'] = joined_data['date'].dt.hour + (joined_data['date'].dt.day-1)*24
    joined_data['weekday'] = joined_data['date'].dt.weekday 
    timeseries = joined_data.groupby(['id','cum_hour']).size()
    max_t = joined_data['cum_hour'].max()+1
    new_index = pd.MultiIndex.from_product([timeseries.index.get_level_values(0).unique(), 
        range(max_t)])
    timeseries = timeseries.reindex(new_index, fill_value=0)
    #save_pickle(timeseries, 'timeseries.pickle')
    
    #timeseries = load_pickle('timeseries.pickle')
    max_t = timeseries.index.get_level_values(1).unique().max()
    count = timeseries.groupby(level=0).sum()

    print 'Compute autocorrelation'
    A_positive = {grid_id: [a if a>0 else 0 for i,a in enumerate(acf(timeseries[grid_id], nlags=max_t, fft=True))] for grid_id in timeseries.index.get_level_values(0).unique()}

    def obj_func(x, T_1=24, T_2=168):
        f = 0.0
        g = [0.0]*1
        for l in range(1, t_max):
            f += (A_positive_l[l] - x[0]*w(l, x[1:]))**2
            g[0] += w(l, x[1:]) 
        g[0] += -1
        fail = 0
        return f, g, fail

    def obj_func_lagrange_relax(x, T_1=24, T_2=168):
        f = 0.0
        g = []
        lambda_ = 0.0
        for l in range(1, t_max):
            f += (A_positive_l[l] - x[0]*w(l, x[1:]))**2
            lambda_ += w(l, x[1:])
        lambda_ += -1
        f += lambda_val*lambda_
        fail = 0
        return f, g, fail

    print 'Loading problem...'
    results = {}
    for c in timeseries.index.get_level_values(0).unique():
        A_positive_l = A_positive[c]
        t_max = max_t
        opt_prob = Optimization('Opt{}'.format(c), obj_func) 
        #opt_prob = Optimization('Opt{}'.format(c), obj_func_lagrange_relax)    
        opt_prob.addVar('x0','c', lower=0, upper=2, value=0.25)
        opt_prob.addVar('x1','c', lower=0, upper=1, value=0.9)
        opt_prob.addVar('x2','c', lower=0, upper=1, value=0.999)
        opt_prob.addVar('x3','c', lower=0, upper=1, value=0.1)
        opt_prob.addVar('x4','c', lower=0, upper=1, value=0.1)
        opt_prob.addObj('f')
        #opt_prob.addCon('g', type='e')
        print 'Solving Opt{}'.format(c)
        alpso = ALPSO()
        alpso.setOption('SwarmSize', 200)
        sol = alpso(opt_prob)
        results[int(c)] = sol[1][1:]

    save_pickle(results, '../data/rhos_dict_louisville.pickle')

    '''
    print 'Solving problem'
    nsga2 = NSGA2()
    #nsga2.setOption('PopSize', 400)
    #nsga2.setOption('maxGen', 200)
    lambda_val = 9.7725862776259942e-2
    sol = nsga2(opt_prob)
    print sol
    print opt_prob.solution(0)
    save_pickle([list(sol[1]), list(A_positive_l)], 'data/acf_dict_nsga2.pickle')

    alpso = ALPSO()
    alpso.setOption('SwarmSize', 500)
    sol = alpso(opt_prob)
    print(sol)
    print(opt_prob.solution(0))
    save_pickle([list(sol[1]), list(A_positive_l)], 'data/acf_dict_alpso.pickle')

    algencan = ALGENCAN()
    sol = algencan(opt_prob)
    print sol
    print opt_prob.solution(0)
    save_pickle([list(sol[0]), list(A_positive_l)], 'data/acf_dict_algencan.pickle')
 
    alhso = ALHSO()
    sol = alhso(opt_prob)
    print sol
    print opt_prob.solution(0)
    save_pickle([list(sol[1]), list(A_positive_l)], 'data/acf_dict_alhso.pickle')

    ksopt = KSOPT()
    sol = ksopt(opt_prob)
    print sol
    print opt_prob.solution(0)
    save_pickle([list(sol[1]), list(A_positive_l)], 'data/acf_dict_ksopt.pickle')

    midaco = MIDACO()  
    sol = midaco(opt_prob)
    print sol
    print opt_prob.solution(0)
    save_pickle([list(sol[1]), list(A_positive_l)], 'data/acf_dict_midaco.pickle')
    '''