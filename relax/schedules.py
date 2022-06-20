import numpy as np
import matplotlib.pyplot as plt

from math import ceil


def to_discrete(x, discrete):
    if discrete:
        return ceil(x) #round(x)
    else:
        return x


class BaseSchedule(object):
    
    def value(self, t):
        raise NotImplementedError
        
    def plot(self, T):       
        t_vec = list(range(T))
        values = list(map(lambda t: self.value(t), t_vec))
        plt.plot(t_vec, values)
        plt.title(f'Schedule plot for {type(self).__name__}')
        plt.show()
        
             
class ConstantSchedule(BaseSchedule):
    
    def __init__(self, start_value, discrete=False):
        
        self.start_value = start_value
        self.discrete = discrete
        
    def value(self, t):
        out = self.start_value
        return to_discrete(out, self.discrete)

    
class ExponentialSchedule(BaseSchedule):
    
    def __init__(self, start_value, gamma, discrete=False):
        
        self.start_value = start_value
        self.gamma = gamma
        self.discrete = discrete
        
    def value(self, t):
        out = self.start_value * (self.gamma**t)
        return to_discrete(out, self.discrete)

    
def linear_interpolation(a, b, T, t):
    return (a * (T - t) + b * t) / T
    

class LinearSchedule(BaseSchedule):
    
    def __init__(self, start_value, end_value, n_steps, discrete=False):
        
        self.start_value = start_value
        self.end_value = end_value
        self.n_steps = n_steps
        self.discrete = discrete
        
    def value(self, t):
        out = None
        if t > self.n_steps:
            out = self.end_value
        else:
            out = linear_interpolation(self.start_value,
                                       self.end_value,
                                       self.n_steps,
                                       t)
        return to_discrete(out, self.discrete)
    

class PiecewiseSchedule(BaseSchedule):
    
    def __init__(self, lr_dict, end_value, discrete=False):
        
        self.start_value = list(lr_dict.keys())[0]
        self.end_value = end_value
        self.lr_dict = lr_dict
        self.max_t = sum(lr_dict.values())
        self.discrete = discrete
        
    def value(self, t):
        out = None
        if t >= self.max_t:
            out = self.end_value
        else:
            csum = 0
            for key, value in self.lr_dict.items():
                if t >= csum and t < csum + value:
                    out = key
                    break
                else:
                    csum += value
        return to_discrete(out, self.discrete)
    
    
class RandomCategoricalSchedule(BaseSchedule):
    
    def __init__(self, support, probs, discrete=True):
        
        self.support = support
        self.probs = probs
        
        self.discrete = True # Always discrete
        self.start_value = self.value(t=0)
        
    def value(self, t):
        
        out = np.random.choice(
            a=self.support,
            p=self.probs,
            size=1
        )[0]
        
        return out
    

class CombinedSchedule(BaseSchedule):
    
    def __init__(self, schedule1, schedule2, agg_func, discrete=False):
        
        self.schedule1 = schedule1
        self.schedule2 = schedule2
        self.agg_func = agg_func
        self.start_value = self.agg_func([self.schedule1.start_value,
                                          self.schedule2.start_value])
        self.discrete = discrete
        
    def value(self, t):
        
        out = self.agg_func([self.schedule1.value(t),
                             self.schedule2.value(t)])
        
        return to_discrete(out, self.discrete)

    
def init_schedule(num_or_schedule, discrete=False):
    
    if not isinstance(num_or_schedule, BaseSchedule):
        return ConstantSchedule(num_or_schedule, discrete=discrete)
    else:
        return num_or_schedule
        