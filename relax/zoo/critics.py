import torch

from torch import nn
from torch.nn import functional as F

from relax.zoo.layers import NoisyLinear


class VMLP(nn.Module):
    
    def __init__(self, obs_dim, nlayers,
                 nunits, activation=nn.Tanh(),
                 out_activation=nn.Identity(),
                 pre_process_module=None):
        
        super(VMLP, self).__init__()
        
        layers = []
        in_size = obs_dim
        
        if pre_process_module is not None:
            layers.append(pre_process_module)
        
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        layers.append(nn.Linear(in_size, 1))
        layers.append(out_activation)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    
    
class VLSTM(nn.Module):
    
    def __init__(self, obs_dim, 
                 seq_len, nlayers_lstm,
                 nunits_lstm, nunits_dense,
                 activation=nn.Tanh(),
                 out_activation=nn.Identity(),
                 pre_process_module=None):
        
        super(VLSTM, self).__init__()
        
        self.pre_process_module = pre_process_module
        self.lstm = nn.LSTM(obs_dim, nunits_lstm, nlayers_lstm)
        self.dense1 = nn.Linear(nunits_lstm, nunits_dense)
        self.act_dense1 = activation
        self.flatten = nn.Flatten()
        self.dense_out = nn.Linear(nunits_dense * seq_len, 1)
        self.act_out = out_activation
        
        
    def forward(self, x):
        
        if self.pre_process_module is not None:
            x = self.pre_process_module(x)
            
        h, _ = self.lstm(x)
        dense1 = self.dense1(h)
        dense1 = self.act_dense1(dense1)
        flatten = self.flatten(dense1)
        dense_out = self.dense_out(flatten)
        values = self.act_out(dense_out)
        return values


class DiscQMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, nlayers,
                 nunits, activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 pre_process_module=None):
        
        super(DiscQMLP, self).__init__()
        
        layers = []
        in_size = obs_dim
        
        if pre_process_module is not None:
            layers.append(pre_process_module)
        
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        layers.append(nn.Linear(in_size, acs_dim))
        layers.append(out_activation)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):        
        y = self.layers(x)     
        return y
    

class DiscNoisyQMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, nlayers,
                 nunits, activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 pre_process_module=None,
                 std_init=0.5):
        
        super(DiscNoisyQMLP, self).__init__()
        
        layers = []
        in_size = obs_dim
        
        if pre_process_module is not None:
            layers.append(pre_process_module)
        
        for _ in range(nlayers):
            layers.append(NoisyLinear(in_size, nunits, std_init=std_init))
            layers.append(activation)
            in_size = nunits
        layers.append(NoisyLinear(in_size, acs_dim, std_init=std_init))
        layers.append(out_activation)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):        
        y = self.layers(x)     
        return y
    
    
class DiscDistributionalQMLP(nn.Module):
    
    def __init__(self, 
                 obs_dim, acs_dim, 
                 nlayers, nunits, 
                 n_atoms, v_min, v_max,
                 activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 pre_process_module=None):
        
        super(DiscDistributionalQMLP, self).__init__()
        
        # Categorical DQN specific params
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.d_z = (v_max-v_min)/(n_atoms-1)
        self.register_buffer('support', torch.Tensor(n_atoms))
        self.support.data.copy_(
            torch.tensor(
                [v_min + i*self.d_z for i in range(n_atoms)]
            )
        )

        self.acs_dim = acs_dim
        self.obs_dim = obs_dim
        
        layers = []
        in_size = obs_dim
        
        if pre_process_module is not None:
            layers.append(pre_process_module)
        
        for _ in range(nlayers):
            layers.append(nn.Linear(in_size, nunits))
            layers.append(activation)
            in_size = nunits
        layers.append(nn.Linear(in_size, acs_dim * n_atoms))
        layers.append(out_activation)
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:        
        out = self.layers(x)
        logits = out.view(-1, self.acs_dim, self.n_atoms)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    
class DiscDistributionalNoisyDuelingQMLP(nn.Module):
    
    def __init__(self, 
                 obs_dim, acs_dim, 
                 nlayers, nunits, 
                 n_atoms, v_min, v_max,
                 activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 pre_process_module=None,
                 std_init=0.5):
        
        super(DiscDistributionalNoisyDuelingQMLP, self).__init__()
        
        # Categorical DQN specific params
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.d_z = (v_max-v_min)/(n_atoms-1)
        self.register_buffer('support', torch.Tensor(n_atoms))
        self.support.data.copy_(
            torch.tensor(
                [v_min + i*self.d_z for i in range(n_atoms)]
            )
        )

        self.acs_dim = acs_dim
        self.obs_dim = obs_dim
        
        layers = []
        in_size = obs_dim
        
        if pre_process_module is not None:
            layers.append(pre_process_module)
        
        for _ in range(nlayers):
            layers.append(NoisyLinear(in_size, nunits, std_init=std_init))
            layers.append(activation)
            in_size = nunits
        
        self.layers = nn.Sequential(*layers)
        
        self.advantages_fc = nn.Sequential(
            NoisyLinear(in_size, acs_dim * n_atoms, std_init=std_init),
        )
        
        self.values_fc = nn.Sequential(
            NoisyLinear(in_size, 1 * n_atoms, std_init=std_init),
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor: 
        
        features = self.layers(x)

        advantages = self.advantages_fc(features)
        values = self.values_fc(features)

        advantages = advantages.view(-1, self.acs_dim, self.n_atoms)
        values = values.view(-1, 1, self.n_atoms)

        logits = values + (advantages - advantages.mean(1, keepdim=True))

        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs


class PreprocessImg(nn.Module):
    
    def forward(self, x):
        # add dimension in case of singular image:
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        # shift from N, H, W, C to N, C, H, W
        x_nchw = x.permute(0, 3, 1, 2).contiguous()
        # scale to 1
        return x_nchw / 255.
    

class AtariQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim):
        
        super(AtariQCNN, self).__init__()
        
        self.net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, acs_dim),
        )
        
    def forward(self, x):
        
        logits = self.net(x)
        
        return logits

    
class AtariDuelingQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim):
        
        super(AtariDuelingQCNN, self).__init__()
        
        self.conv_net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.advantages_fc = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, acs_dim),
        )
        
        self.values_fc = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        
    def forward(self, x):
        
        features = self.conv_net(x)
        advantages = self.advantages_fc(features)
        values = self.values_fc(features)
        
        qvalues = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return qvalues
    
    
class AtariNoisyQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim, std_init=0.5):
        
        super(AtariNoisyQCNN, self).__init__()
        
        self.net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            NoisyLinear(3136, 512, std_init=std_init),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            NoisyLinear(512, acs_dim, std_init=std_init),
        )
        
    def forward(self, x):
        
        logits = self.net(x)
        
        return logits
    

class AtariNoisyDuelingQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim, std_init=0.5):
        
        super(AtariNoisyDuelingQCNN, self).__init__()
        
        self.conv_net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.advantages_fc = nn.Sequential(
            NoisyLinear(3136, 512, std_init=std_init),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            NoisyLinear(512, acs_dim, std_init=std_init),
        )
        
        self.values_fc = nn.Sequential(
            NoisyLinear(3136, 512, std_init=std_init),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            NoisyLinear(512, 1, std_init=std_init),
        )
        
    def forward(self, x):
        
        features = self.conv_net(x)
        advantages = self.advantages_fc(features)
        values = self.values_fc(features)
        
        qvalues = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        return qvalues
    
    
class AtariDistributionalQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim,
                 n_atoms, v_min, v_max):
        
        super(AtariDistributionalQCNN, self).__init__()
        
        # Categorical DQN specific params
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.d_z = (v_max-v_min)/(n_atoms-1)
        self.register_buffer('support', torch.Tensor(n_atoms))
        self.support.data.copy_(
            torch.tensor(
                [v_min + i*self.d_z for i in range(n_atoms)]
            )
        )

        self.acs_dim = acs_dim
        
        self.net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, acs_dim * n_atoms),
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:        
        out = self.net(x)
        logits = out.view(-1, self.acs_dim, self.n_atoms)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    
class AtariDistributionalDuelingQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim,
                 n_atoms, v_min, v_max):
        
        super(AtariDistributionalDuelingQCNN, self).__init__()
        
        # Categorical DQN specific params
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.d_z = (v_max-v_min)/(n_atoms-1)
        self.register_buffer('support', torch.Tensor(n_atoms))
        self.support.data.copy_(
            torch.tensor(
                [v_min + i*self.d_z for i in range(n_atoms)]
            )
        )

        self.acs_dim = acs_dim
        
        self.conv_net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.advantages_fc = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, acs_dim * n_atoms),
        )
        
        self.values_fc = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, 1 * n_atoms),
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor: 
        
        features = self.conv_net(x)
        
        advantages = self.advantages_fc(features)
        values = self.values_fc(features)
        
        advantages = advantages.view(-1, self.acs_dim, self.n_atoms)
        values = values.view(-1, 1, self.n_atoms)
        
        logits = values + (advantages - advantages.mean(1, keepdim=True))
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    
class AtariDistributionalNoisyDuelingQCNN(nn.Module):
    
    def __init__(self, in_channels, acs_dim,
                 n_atoms, v_min, v_max,
                 std_init=0.5):
        
        super(AtariDistributionalNoisyDuelingQCNN, self).__init__()
        
        # Categorical DQN specific params
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.d_z = (v_max-v_min)/(n_atoms-1)
        self.register_buffer('support', torch.Tensor(n_atoms))
        self.support.data.copy_(
            torch.tensor(
                [v_min + i*self.d_z for i in range(n_atoms)]
            )
        )

        self.acs_dim = acs_dim
        
        self.conv_net = nn.Sequential(
            PreprocessImg(),
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        self.advantages_fc = nn.Sequential(
            NoisyLinear(3136, 512, std_init=std_init),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            NoisyLinear(512, acs_dim * n_atoms, std_init=std_init),
        )
        
        self.values_fc = nn.Sequential(
            NoisyLinear(3136, 512, std_init=std_init),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            NoisyLinear(512, 1 * n_atoms, std_init=std_init),
        )
        
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor: 
        
        features = self.conv_net(x)
        
        advantages = self.advantages_fc(features)
        values = self.values_fc(features)
        
        advantages = advantages.view(-1, self.acs_dim, self.n_atoms)
        values = values.view(-1, 1, self.n_atoms)
        
        logits = values + (advantages - advantages.mean(1, keepdim=True))
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    
class ContQMLP(nn.Module):
    
    def __init__(self, obs_dim, acs_dim, 
                 hidden1=400, hidden2=300,
                 activation=nn.ReLU(),
                 out_activation=nn.Identity(),
                 init_w=3e-3,
                 pre_process_module=None):
        
        super(ContQMLP, self).__init__()
        
        self.pre_process_module = pre_process_module
        self.fc1 = nn.Linear(obs_dim+acs_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.activation = activation
        self.out_activation = out_activation
        self.acs_dim = acs_dim
        
    def forward(self, obs, acs):
        
        if self.pre_process_module is not None:
            obs = self.pre_process_module(obs)
        
        out = self.fc1(
            torch.cat(
                [torch.flatten(obs, start_dim=1), acs], 
                dim=-1
            )
        )
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        out = self.out_activation(out)
        
        return out
    