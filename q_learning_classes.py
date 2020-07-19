import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math
import random
from collections import namedtuple
from itertools import count


Experience = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        a = random.sample(self.memory, batch_size)
        return a

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self,start,end,decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear_1 = nn.Linear(21,64)
        self.linear_2 = nn.Linear(64,64)
        #self.linear_3 = nn.Linear(64,64)
        self.linear_out = nn.Linear(64,4)

    def forward(self, t):
        #t = t.flatten(start_dim=1)
        t = torch.tanh(self.linear_1(t))
        t = torch.tanh(self.linear_2(t))
        #t = torch.relu(self.linear_3(t))
        t = self.linear_out(t)
        return t

class Agent():
    def __init__(self, strategy, num_actions):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.random_action = None

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1

        if rate > random.random():
            self.random_action = 1
            return torch.tensor([random.randrange(self.num_actions)],dtype=torch.double).to(torch.device("cpu"),non_blocking=True) # explore
        else:
            self.random_action = 0
            with torch.no_grad():
                return torch.tensor([policy_net(state.flatten()).argmax(dim=0)],dtype=torch.double).to(torch.device("cpu"),non_blocking=True) # exploit
