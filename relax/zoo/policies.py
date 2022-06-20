import torch

from torch import nn
from torch.nn import functional as F
from torch.distributions import MultivariateNormal, Categorical

from relax.zoo.distributions import TanhNormal


class CategoricalMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, nlayers,
                 nunits, activation=nn.Tanh(),
                 out_activation=nn.Identity()):
        
        super(CategoricalMLP, self).__init__()
        
        layers = []
        in_size = obs_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        layers.append(nn.Linear(in_size, acs_dim))
        layers.append(out_activation)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        
        logits = self.layers(x)
        y = Categorical(probs=F.softmax(logits, dim=-1))
        
        return y
    

class NormalMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, nlayers,
                 nunits, activation=nn.Tanh(),
                 out_activation=nn.Identity(),
                 acs_scale=1, acs_bias=0,
                 init_log_std=0.0):
        
        super(NormalMLP, self).__init__()
        
        layers = []
        in_size = obs_dim
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        
        self.layers = nn.Sequential(*layers)
        
        self.mean_layer = nn.Sequential(
            nn.Linear(nunits, acs_dim),
            out_activation
        )
        
        self.log_stds = nn.Parameter(
            torch.ones(acs_dim) * init_log_std
        )
        
        self.acs_scale = acs_scale
        self.acs_bias = acs_bias

    def forward(self, x):
        
        x_seq = self.layers(x)
        means = self.mean_layer(x_seq)
        means = means * self.acs_scale + self.acs_bias
        cov = torch.diag_embed(torch.exp(self.log_stds))
            
        y = MultivariateNormal(loc=means,
                               covariance_matrix=cov)
        return y
    

class NormalLSTM(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, nlayers_lstm,
                 nunits_lstm, nunits_dense, activation=nn.Tanh(),
                 out_activation=nn.Identity()):
        
        super(NormalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(obs_dim, nunits_lstm, nlayers_lstm)
        self.dense1 = nn.Linear(nunits_lstm, nunits_dense)
        self.act_dense1 = activation
        self.dense_out = nn.Linear(nunits_dense, acs_dim)
        self.act_out = out_activation
        
        self.log_stds = nn.Parameter(
            torch.zeros(acs_dim)
        )    
        
    def forward(self, x):
        #if len(x.shape) < 3:
        #    x = x.unsqueeze(0)
        h, _ = self.lstm(x)
        pooled, _ = torch.max(h, 1)
        dense1 = self.dense1(pooled)
        dense1 = self.act_dense1(dense1)
        dense_out = self.dense_out(dense1)
        means = torch.squeeze(self.act_out(dense_out))
        cov = torch.diag_embed(torch.exp(self.log_stds))
        
        y = MultivariateNormal(loc=means,
                               covariance_matrix=cov)
        
        return y
    

class DeterministicMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, 
                 hidden1=400, hidden2=300,
                 activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 acs_scale=1.0, acs_bias=0.0):
        
        super(DeterministicMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(obs_dim, hidden1))
        layers.append(activation)
        layers.append(nn.Linear(hidden1, hidden2))
        layers.append(activation)
        layers.append(nn.Linear(hidden2, acs_dim))
        layers.append(out_activation)
        self.layers = nn.Sequential(*layers)
        self.acs_dim = acs_dim
        self.acs_scale = acs_scale
        self.acs_bias = acs_bias
        
    def forward(self, obs):
        flat_obs = torch.flatten(obs, start_dim=1)
        acs = self.layers(flat_obs)
        return acs * self.acs_scale + self.acs_bias
    

class TanhNormalMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, 
                 hidden1=400, hidden2=300,
                 acs_scale=1, acs_bias=0,
                 activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 init_w=3e-3, 
                 min_log_std=-20, 
                 max_log_std=2):
        
        super(TanhNormalMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(obs_dim, hidden1))
        layers.append(activation)
        layers.append(nn.Linear(hidden1, hidden2))
        layers.append(activation)
        self.hidden_layers = nn.Sequential(*layers)
        
        self.mean_layer_linear = nn.Linear(hidden2, acs_dim)
        self.mean_layer_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_layer_linear.bias.data.uniform_(-init_w, init_w)
        self.mean_layer = nn.Sequential(
            self.mean_layer_linear,
            out_activation
        )
        
        self.log_std_layer_linear = nn.Linear(hidden2, acs_dim)
        self.log_std_layer_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_layer_linear.bias.data.uniform_(-init_w, init_w)
        self.log_std_layer = nn.Sequential(
            self.log_std_layer_linear,
            out_activation
        )
        
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.acs_scale = acs_scale
        self.acs_bias = acs_bias

    def forward(self, x):
        
        x_hidden = self.hidden_layers(torch.flatten(x, start_dim=1))
        means = self.mean_layer(x_hidden)
        log_stds = self.log_std_layer(x_hidden)
        
        y = TanhNormal(
            loc=means,
            log_scale=log_stds,
            acs_scale=self.acs_scale,
            acs_bias=self.acs_bias,
            min_log_std=self.min_log_std,
            max_log_std=self.max_log_std
        )
        
        return y
