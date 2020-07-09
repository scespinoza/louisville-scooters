import time
from collections import deque
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SubActor(nn.Module):
    def __init__(self, neurons=16, state_size=6):
        super(SubActor, self).__init__()
        self.gru = nn.GRU(state_size, neurons, batch_first=True)
        self.fc1 = nn.Linear(neurons, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
    def forward(self, x):
        x, h = self.gru(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = nn.ReLU()(self.fc3(x))
        return x

class ActorNetwork(nn.Module):
    def __init__(self, n_subactors=100, neurons=16, state_size=6):
        super(ActorNetwork, self).__init__()
        self.n_subactors = n_subactors
        for i in range(n_subactors):
            setattr(self, 'subactor{}'.format(i), SubActor(neurons=neurons, state_size=state_size))

    def forward(self, x):
        batch, t, nzones, state_size = x.size()
        a = []
        for i in range(self.n_subactors):
            sa = getattr(self, 'subactor{}'.format(i))(x[:, :, i])
            a.append(sa)
        return torch.stack(a).view(-1, t, nzones)

class SimpleSubActor(nn.Module):
    def __init__(self, input_size=16):
        super(SimpleSubActor, self).__init__()
        self.fc1 = nn.Linear(input_size,512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleActor(nn.Module):
    def __init__(self, state_size=6, gru_out=64, nzones=100):
        super(SimpleActor, self).__init__()
        self.nzones = nzones
        self.gru = nn.GRU(state_size, gru_out, batch_first=True)
        self.subactor = SimpleSubActor(input_size=gru_out)

    def forward(self, x):
        batch, t, nzones, state_size = x.size()
        a = []
        for i in range(self.nzones):
            sx, h = self.gru(x[:, :, i])
            sa = self.subactor(sx)
            a.append(sa)
        return torch.stack(a).view(-1, t, nzones)
    
class LocalizedModule(nn.Module):
    def __init__(self, state_size=6):
        super(LocalizedModule, self).__init__()
        # Input is state_size * neighbors + price (action)
        self.fc1 = nn.Linear((state_size + 1) * 5 , 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256 , 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x        

class SubCritic(nn.Module):
    def __init__(self, neurons=16, state_size=6):
        super(SubCritic, self).__init__()
        self.gru = nn.GRU((state_size + 1) * 5, neurons, batch_first=True)
        self.fc1 = nn.Linear(neurons, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)

    def forward(self, x):
        x, h = self.gru(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x[:, -1, :]

class CriticNetwork(nn.Module):
    def __init__(self, n_subcritics=100, neurons=16, state_size=6):
        super(CriticNetwork, self).__init__()
        self.n_subcritics = n_subcritics
        for i in range(n_subcritics):
            setattr(self, 'lm{}'.format(i), LocalizedModule())
            setattr(self, 'subcritic{}'.format(i), SubCritic(neurons=neurons, state_size=state_size))
        
    def forward(self, x):
        for i in range(self.n_subcritics):
            neighbors = (i, i - 10, i + 1, i + 10, i - 1)
            xi = []
            for n in neighbors:
                if n >= 0 and n < self.n_subcritics:
                    xi.append(x[:, :, n])
                else:
                    xi.append(torch.zeros_like(x[:,:,i]))
            xi = torch.cat(xi, dim=2)
            f = []
            q = []
            
            f.append(getattr(self, 'lm{}'.format(i))(xi[:, -1, :].view(-1, 5 * 7)))
            q.append(getattr(self, 'subcritic{}'.format(i))(xi))
        f = torch.cat(f)
        q = torch.cat(q)
        
        return torch.sum(f + q, dim=1)

class SimpleSubCritic(nn.Module):
    def __init__(self, input_size=16):
        super(SimpleSubCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x[:, -1, :]

class SimpleCritic(nn.Module):
    def __init__(self, gru_out=32, state_size=6, nzones=100):
        super(SimpleCritic, self).__init__()
        self.nzones = nzones
        self.gru = nn.GRU((state_size + 1) * 5, gru_out, batch_first=True)
        self.subcritic = SimpleSubCritic(input_size=gru_out)
        self.lm = LocalizedModule(state_size=6)

    def forward(self, x):
        for i in range(self.nzones):
            neighbors = (i, i - 10, i + 1, i + 10, i - 1)
            xi = []
            for n in neighbors:
                if n >= 0 and n < self.nzones:
                    xi.append(x[:, :, n])
                else:
                    xi.append(torch.zeros_like(x[:,:,i]))
            xi = torch.cat(xi, dim=2)
            f = []
            q = []
            
            f.append(self.lm(xi[:, -1, :].view(-1, 5 * 7)))
            xq, h = self.gru(xi)
            q.append(self.subcritic(xq))
        f = torch.cat(f)
        q = torch.cat(q)
        return torch.sum(f + q, dim=1)


class HRP(nn.Module):
    def __init__(self, learning_rate=1e-4, target_model=False):
        super(HRP, self).__init__()
        self.an = SimpleActor()
        self.cn = SimpleCritic()
        self.critic_criterion = nn.MSELoss()
        self.discount_rate = 0.99
        self.learning_rate = 1e-4
        self.tau = 0.2
        self.critic_optimizer = torch.optim.Adam(self.cn.parameters(), lr=self.learning_rate)
        self.actor_optimizer = torch.optim.Adam(self.an.parameters(), lr=self.learning_rate)
        if not target_model:
            self.target_model = HRP(target_model=True)

    def forward(self, x):
        p = self.an(x)
        xc = torch.cat([x, p.view(-1, 2, 100, 1)], dim=-1)
        q = self.cn(xc)
        return p[:, -1], q.view(-1, 1)
    
    def critic_loss(self, batch):
        p, q = self.forward(batch['state'])
        with torch.no_grad():
            p_next, q_next = self.target_model(batch['next_state'])
            y = batch['reward'] + self.discount_rate * q_next
        distance = (y - q).detach().numpy().mean()
        return self.critic_criterion(y,q), distance

    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def critic_step(self, batch):
        self.critic_optimizer.zero_grad()
        batch_loss, _ = self.critic_loss(batch)
        batch_loss.backward()
        self.critic_optimizer.step()
        return batch_loss.detach().numpy()

    def actor_gradient(self, batch):
        p = self.an(batch['state'])
        xc = torch.cat([batch['state'], p.view(-1, 2, 100, 1)], dim=-1)
        q = self.cn(xc)
        return torch.mean(-1 * q.squeeze())
        
    def actor_step(self, batch):
        grad = self.actor_gradient(batch)
        self.actor_optimizer.zero_grad()
        grad.backward()
        self.actor_optimizer.step()
        return grad.detach().numpy()

    def train_step(self, batch):
        batch_loss = self.critic_step(batch)
        q_val = self.actor_step(batch)
        self.soft_update()
        return batch_loss, q_val
                    
    def sample_batch(self, n=16):
        torch.manual_seed(0)
        return {
            'state': torch.randn(n, 2, 100, 6),
            'next_state': torch.randn(n, 2, 100, 6),
            'reward': torch.randn(n, 1),
            'action': torch.randn(n, 100)
        }
    def train(self, iter=24):
        history = {
            'batch_loss': [],
            'q_val': []
        }
        start = time.time()
        for i in range(iter):
            print('Training step {}/{}'.format(i + 1, iter))
            batch = self.sample_batch()
            batch_loss, q_val = self.train_step(batch)
            print('Batch Loss: {:.2f}'.format(batch_loss))
            history['batch_loss'].append(batch_loss)
            history['q_val'].append(q_val)
        history = {metric: np.stack(history[metric]) for metric in history}
        print('Elapsed Time : {:.2f}'.format(time.time() - start))
        return history

    def save_target_model(self, filename='weights/hrp.pth'):
        torch.save(self, filename)

class Agent:

    def __init__(self, name, model, buffer_length):
        self.name = name
        self.model = model
        self.experience_buffer = deque(maxlen=buffer_length)
        self.random_exploration_process = np.random.rand
        self.history = {
            'rewards': [],
            'dqn_loss': [],
            'batch_loss': [],
            'distance': []
        }

    def store_transition(self, transition):
        self.experience_buffer.append(transition)

    def sample_minibatch(self, batch_size):
        batch = random.sample(self.experience_buffer, batch_size)
        return [[sample[i] for sample in batch] for i in range(4)]

    def train_minibatch(self, batch_size=32):
        state, action, reward, next_state = self.sample_minibatch(batch_size)
        x = {'state': torch.from_numpy(np.concatenate(state)),
            'action': torch.from_numpy(np.concatenate(action)), 
            'reward': torch.from_numpy(np.array(reward)).view(-1, 1),
            'next_state': torch.from_numpy(np.concatenate(next_state))}
        self.model.train_step(x)
        return self.model.critic_loss(x)

    def get_action(self, state):
        return self.model.an(state).detach().numpy()[:, -1].reshape(10, 10)

    def act(self, environment, episode=0):
        state  = environment.get_state()
        action = self.get_action(torch.from_numpy(state))
        scale = 5 * (0.99 ** episode)
        noise = np.random.normal(size=action.shape, scale=scale)
        action = action + noise
        next_state, reward = environment.perform_action(action)
        self.store_transition((state, action, reward, next_state))
        return reward
    
    def warmup_phase(self, environment, iterations=10):
        for i in range(iterations):
            environment.reset()
            episode_rewards = []
            for t in range(environment.timesteps):
                print('\n')
                print('Warmup Iteration {}/{}, timestep {}'.format(i + 1, iterations, t + 1))
                reward = self.act(environment)
                print('Reward: {:.2f}'.format(reward))
                episode_rewards.append(self.model.discount_rate*reward)
            self.history['rewards'].append(np.sum(episode_rewards))

    def train(self, environment, episodes=100, warmup_iterations=1):
        self.episodes = episodes
        self.warmup_iterations = warmup_iterations
        print('Starting Warmup Phase')
        warmup_start = time.time()
        self.warmup_phase(environment, iterations=warmup_iterations)
        print('Finishing Warmup Phase: {:.2f}s'.format(time.time() - warmup_start))
        print('Starting Training')
        training_start = time.time()
        for e in range(episodes):
            environment.reset()
            episode_rewards = []
            for t in range(environment.timesteps):
                print('Episode {}, timestep {}'.format(e + 1, t + 1))
                reward = self.act(environment, episode=e)
                print('Reward: {:.2f}, '.format(reward), end='')  
                episode_rewards.append(self.model.discount_rate*reward)         
                batch_loss, distance = self.train_minibatch()
                print('Batch Loss: {:.2f}'.format(batch_loss.detach().numpy()))
                self.history['distance'].append(distance)
                self.history['batch_loss'].append(batch_loss.detach().numpy())
            self.history['rewards'].append(np.sum(episode_rewards))
            self.history['dqn_loss'].append(self.get_q_loss())
        print('Finishing Training: {:.2f}s'.format(time.time() - training_start))

    def get_q_loss(self):
        state, action, reward, next_state = [[sample[i] for sample in self.experience_buffer] for i in range(4)]
        return self.model.critic_loss({'state': torch.from_numpy(np.concatenate(state)),
                                    'action': torch.from_numpy(np.concatenate(action)), 
                                    'reward': torch.from_numpy(np.array(reward)).view(-1, 1),
                                    'next_state': torch.from_numpy(np.concatenate(next_state))})[0].detach().numpy()
    def save_target_model(self):
        self.model.save_target_model()

    def load_trained(self, filename = 'weights/hrp.pth'):
        self.model = torch.load(filename)
        self.model.eval()
        return self

 

