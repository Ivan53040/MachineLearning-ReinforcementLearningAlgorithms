import math

import torch.nn as nn

# Define the network structure - in this case 2 hidden layers (CartPole can be solved faster with a single hidden layer)
class DqnNetTwoLayers(nn.Module):
  def __init__(self, obs_size, hidden_size, hidden_size2, n_actions):
    super(DqnNetTwoLayers, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, hidden_size2),
      nn.ReLU(),
      nn.Linear(hidden_size2, n_actions)
    )

  def forward(self, x):
    return self.net(x.float())

class DqnNetSingleLayer(nn.Module):
  def __init__(self, obs_size, hidden_size, n_actions):
    super(DqnNetSingleLayer, self).__init__()
    self.net = nn.Sequential(
      nn.Linear(obs_size, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, n_actions)
    )

  def forward(self, x):
    return self.net(x.float())

class DuellingDqn(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
      super(DuellingDqn, self).__init__()
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # we have 2 nets now - one for values and one for advantage (i.e. the difference each action causes)
      # with 2 layers it doesn't converge!!!!!!!!!!!!!!!!!
      self.value_net = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        # nn.ReLU(),
        # nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # careful - the output here is just a single state value
        nn.Linear(hidden_size, 1)
      )
      self.advantage_net = nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        # nn.ReLU(),
        # nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions)
      )

    def forward(self, x):
      # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      # Q = value + advantage - advantage.mean()
      value = self.value_net(x.float())
      advantage = self.advantage_net(x.float())
      return value + advantage - advantage.mean()

def epsilon_by_frame(frame_idx, params):
  return params['epsilon_final'] + (params['epsilon_start'] - params['epsilon_final']) * math.exp(-1.0 * frame_idx / params['epsilon_decay'])

def alpha_sync(net, tgt_net, alpha):
  assert isinstance(alpha, float)
  assert 0.0 < alpha <= 1.0
  state = net.state_dict()
  tgt_state = tgt_net.state_dict()
  for k, v in state.items():
    tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
  tgt_net.load_state_dict(tgt_state)

