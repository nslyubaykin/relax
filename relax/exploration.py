import numpy as np

from relax.schedules import init_schedule


class ActionAlterExploration():
    
    def reset_state(self):
        pass
        
    def save_state(self):
        return None
    
    def load_state(self, state):
        pass
    
    def get_logs(self):
        return {}


class EpsilonGreedy(ActionAlterExploration):
    
    def __init__(self, eps):
        
        self.eps = init_schedule(eps)
        self.global_step = 0
        
    def get_action(self, logits):
        
        if len(logits.shape) > 1:
            n_acs = logits.shape[0]
        else:
            n_acs = None
            
        eps_mask = np.random.random(n_acs) < self.eps.value(self.global_step)
        
        random_acs = np.random.random(logits.shape).argmax(-1)
        critic_acs = logits.argmax(-1)
        
        out_acs = np.where(eps_mask, random_acs, critic_acs)
        
        if n_acs is None:
            out_acs = int(out_acs)
            
        return out_acs
    
    def schedules_step(self):
        self.global_step += 1
        
    def get_logs(self):
        logs = {}
        pr = type(self).__name__
        logs[f'{pr}_eps'] = self.eps.value(self.global_step)
        logs[f'{pr}_global_step'] = self.global_step
        return logs

    
class OrnsteinUhlenbeck(ActionAlterExploration):
    
    def __init__(self, 
                 theta,
                 sigma, 
                 dim, 
                 mu=0., 
                 dt=1e-2, 
                 x0=None,
                 n_random_steps=None,
                 min_acs=None,
                 max_acs=None):
        
        self.theta = init_schedule(theta)
        self.sigma = init_schedule(sigma)
        
        self.dt = dt
        self.mu = mu
        
        self.x0 = x0
        self.dim = dim
        
        self.reset_state()
        assert self.x_prev.shape[0] == self.dim
        
        self.global_step = 0
        self.counter = 0
        self.n_random_steps = n_random_steps
        
        self.min_acs = min_acs
        self.max_acs = max_acs
        
    def sample(self):
        x = self.x_prev \
            + self.theta.value(self.global_step) * (self.mu - self.x_prev) * self.dt \
            + self.sigma.value(self.global_step) * np.sqrt(self.dt) * np.random.normal(size=self.dim)
        self.x_prev = x
        return x
    
    def reset_state(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.dim)
        
    def save_state(self):
        return self.x_prev
    
    def load_state(self, state):
        if state is not None:
            self.x_prev = state
            
    def get_action(self, acs: np.ndarray) -> np.ndarray:
        rand_acs = float(self.n_random_steps is not None and self.global_step <= self.n_random_steps)
        out_acs = acs * (1-rand_acs) + self.sample()
        if self.min_acs is not None or self.max_acs is not None:
            out_acs = np.clip(out_acs, a_min=self.min_acs, a_max=self.max_acs)
        return out_acs
    
    def get_logs(self):
        logs = {}
        pr = type(self).__name__
        logs[f'{pr}_sigma'] = self.sigma.value(self.global_step)
        logs[f'{pr}_theta'] = self.theta.value(self.global_step)
        logs[f'{pr}_global_step'] = self.global_step
        return logs
        
    def schedules_step(self):
        self.global_step += 1
        
        
class RandomNormal(ActionAlterExploration):
    
    def __init__(self,
                 sigma, 
                 mu=0.,
                 n_random_steps=None,
                 min_acs=None,
                 max_acs=None):
        
        self.sigma = init_schedule(sigma)
        self.mu = init_schedule(mu)
        
        self.global_step = 0
        self.n_random_steps = n_random_steps
        
        self.min_acs = min_acs
        self.max_acs = max_acs
        
    def schedules_step(self):
        self.global_step += 1
        
    def get_action(self, acs: np.ndarray) -> np.ndarray:
        sigma = self.sigma.value(self.global_step)
        mu = self.mu.value(self.global_step)
        noise = np.random.normal(mu, sigma, acs.shape)
        rand_acs = float(self.n_random_steps is not None and self.global_step <= self.n_random_steps)
        out_acs = acs * (1-rand_acs) + noise
        if self.min_acs is not None or self.max_acs is not None:
            out_acs = np.clip(out_acs, a_min=self.min_acs, a_max=self.max_acs)
        return out_acs
    
    def get_logs(self):
        logs = {}
        pr = type(self).__name__
        logs[f'{pr}_sigma'] = self.sigma.value(self.global_step)
        logs[f'{pr}_global_step'] = self.global_step
        return logs
    