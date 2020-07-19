import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from q_learning_classes import *
from snake_game import *
import time

def extract_tensors(experiences):
    batch = Experience(*zip(*experiences))

    t1 = torch.cat(batch.state)
    t2 = torch.cat(batch.action)
    t3 = torch.cat(batch.next_state)
    t4 = torch.cat(batch.reward)

    return (t1,t2,t3,t4)

class QValues():
    @staticmethod
    def get_current(policy_net,states,actions):
        x = policy_net(states).gather(dim=1,index=torch.tensor(actions,dtype=torch.int64).to(torch.device("cpu"),non_blocking=True).unsqueeze(-1)).to(torch.device("cpu"),non_blocking=True)
        return x

    @staticmethod
    def get_next(target_net, next_states):
        final_state_locations = next_states.squeeze(0) \
            .max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(torch.device("cpu"),non_blocking=True)
        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
        return values

game = SnakeGame(300)
#-----------------------------
batch_size = 1024
gamma = 0.9
eps_start = 1
eps_end = 0.01
eps_decay = 0.002
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 2000

strategy = EpsilonGreedyStrategy(eps_start,eps_end,eps_decay)
agent = Agent(strategy, 4)
memory = ReplayMemory(memory_size)

policy_net = DQN().to(torch.device("cpu"),non_blocking=True)
target_net = DQN().to(torch.device("cpu"),non_blocking=True)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
criterion = nn.MSELoss()
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

for episode in range(num_episodes):
    reward_info = 0
    print("Epoch:",episode,"|","Score:",game.score)
    game.game_reset()
    state = game.get_current_state_2()

    for timestep in count():
        game.run_game()
        action = agent.select_action(state, policy_net)
        game.take_action(int(action.item()))
        reward = torch.tensor(game.reward).unsqueeze(-1).to(torch.device("cpu"),non_blocking=True)
        reward_info += game.reward
        next_state = game.get_current_state_2()
        memory.push(Experience(state,action,next_state,reward))
        state = next_state

        if memory.can_provide_sample(batch_size):
            experiences = memory.sample(batch_size)
            states, actions, next_states, rewards = extract_tensors(experiences)
            current_q_values = QValues.get_current(policy_net, states.view(batch_size,21), actions)
            next_q_values = QValues.get_next(target_net,next_states.view(batch_size,21))
            target_q_values = (next_q_values*gamma) + rewards

            loss = criterion(current_q_values.flatten(), target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if game.done and agent.random_action:
            print("Death by random action taken")
        if game.done:
            break

    if episode % target_update == 0:
        print("Weights updated")
        target_net.load_state_dict(policy_net.state_dict())
