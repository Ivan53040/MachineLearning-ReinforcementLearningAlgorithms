import collections
from collections import namedtuple
import numpy as np
import torch

Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])
ExperienceImageHistory = collections.namedtuple('ExperienceImageHistory', field_names=['history', 'state', 'action', 'reward', 'done', 'new_state'])

class ExperienceBuffer():
  def __init__(self, capacity, device):
    self.buffer = collections.deque(maxlen=capacity)
    self.device = device
    self.size = 0
    self.capacity = capacity

  def __len__(self):
    return len(self.buffer)

  def append(self, experience):
    self.buffer.append(experience)
    if self.size < self.capacity:
      self.size += 1

  def sample(self, batch_size):
    indices = np.random.choice(self.size, batch_size, replace=False)
    states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

    states_input = states if isinstance(states, list) else np.array(states)
    next_states_input = next_states if isinstance(next_states, list) else np.array(next_states)

    return torch.tensor(states_input, dtype=torch.float).to(self.device), \
           torch.tensor(np.array(actions)).to(self.device), \
           torch.tensor(np.array(rewards, dtype=np.float32)).to(self.device), \
           torch.tensor(np.array(dones, dtype=np.uint8)).to(self.device), \
           torch.tensor(next_states_input, dtype=torch.float).to(self.device)

class ExperienceBufferWithHistory(ExperienceBuffer):
  def __init__(self, capacity):
    super().__init__(capacity)

  def sample(self, batch_size):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
    # states and next_states are already np arrays - they were pre-processed through RNN
    return states, \
           np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           next_states

class ExperienceBufferImageHistory(ExperienceBuffer):
  def __init__(self, capacity):
    super().__init__(capacity)

  def sample(self, batch_size):
    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
    histories, states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
    # states and next_states are already np arrays - they were pre-processed through RNN
    return np.array(histories), \
           np.array(states), \
           np.array(actions), \
           np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), \
           np.array(next_states)

def pad_with_zeros(histories, to_size, pad_value = -1000):
  if len(histories) < to_size:
    padding = (np.zeros(len(histories[0])) + pad_value).tolist()
    for _ in range(to_size - len(histories)):
      histories.insert(0, padding)

  return histories