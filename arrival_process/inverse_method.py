# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 19:37:10 2017

@author: Sebastian
"""
import math
from numpy.random import RandomState

class Inverse_Method(object):
    def __init__(self):
        pass

    def integrate(self, func):
        func[0]['a'], func[0]['b'] = func[0]['rate'], 0
        cumulative = func[0]['a']*func[0]['ub']
        for interval in func[1:]:
            interval['a'] = interval['rate']
            interval['b'] = cumulative - interval['lb']*interval['rate']
            cumulative = cumulative + (interval['ub'] - interval['lb'])*interval['a']
    
    def invert(self, func):
        def eval_bound(interval, bound):
            return interval[bound]*interval['a'] + interval['b']
        return [{'lb':eval_bound(interval, 'lb'), 'ub':eval_bound(interval, 'ub'), 
            'a':1/interval['a'], 'b':-interval['b']/interval['a']} for interval in func]

    def sample(self, func, seed):
        self.integrate(func)
        self.inverted_func = self.invert(func)
        self.random_generator = RandomState()
        self.random_generator.seed(seed)

        def eval_inverse(value):
            for interval in self.inverted_func:
                if (value<=interval['ub']) & (value>=interval['lb']):
                    return value*interval['a'] + interval['b']

        s_n, prev_S_n = self.random_generator.exponential(scale=1), 0
        interarrival_times = []
        while s_n < self.inverted_func[-1]['ub']:
            S_n = eval_inverse(s_n)
            interarrival_times.append(S_n - prev_S_n)
            prev_S_n = S_n
            s_n = s_n + self.random_generator.exponential(scale=1)
        return interarrival_times
        
if __name__ == '__main__':

    func=[{'lb':0, 'ub':4, 'rate':2}, {'lb':4, 'ub':6, 'rate':3}]
    inv = Inverse_Method(func)
    x = inv.sample() 
    print(x)