import time
import pickle
from collections import deque
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(42)

def init_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)
        m.bias.data.fill_(0.0)

class SubActor(nn.Module):
    def __init__(self, neurons=32, state_size=6, max_action=3, min_action=0):
        super(SubActor, self).__init__()
        self.max_action = max_action
        self.min_action = min_action
        self.gru = nn.GRU(state_size, neurons, batch_first=True)
        self.fc1 = nn.Linear(neurons, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
        self.apply(init_uniform)
    def forward(self, x):
        x, h = self.gru(x)
        x = nn.ReLU()(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.max_action * (nn.Sigmoid()(self.fc3(x)) - 0.5)
        return x

class ActorNetwork(nn.Module):
    def __init__(self,min_action, max_action, n_subactors=100, neurons=64, state_size=6):
        super(ActorNetwork, self).__init__()
        self.n_subactors = n_subactors
        for i in range(n_subactors):
            setattr(self, 'subactor{}'.format(i), SubActor(min_action=min_action, max_action=max_action,neurons=neurons, state_size=state_size))

    def forward(self, x):
        batch, t, nzones, state_size = x.size()
        a = []
        for i in range(self.n_subactors):
            sa = getattr(self, 'subactor{}'.format(i))(x[:, :, i])
            a.append(sa)
        return torch.stack(a).view(-1, t, nzones)

class SimpleSubActor(nn.Module):
    def __init__(self, neurons=512, input_size=16):
        super(SimpleSubActor, self).__init__()
        self.fc1 = nn.Linear(input_size,neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
        self.scale = torch.nn.Parameter(torch.zeros(1))
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
            sa = self.subactor(nn.ReLU()(sx))
            a.append(sa)
        return torch.stack(a).view(-1, t, nzones)
    
class LocalizedModule(nn.Module):
    def __init__(self, neurons=128, state_size=6):
        super(LocalizedModule, self).__init__()
        # Input is state_size * neighbors + price (action)
        self.fc1 = nn.Linear((state_size + 1) * 5 , neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
        self.apply(init_uniform)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x        

class SubCritic(nn.Module):
    def __init__(self, neurons=64, state_size=6):
        super(SubCritic, self).__init__()
        self.gru = nn.GRU((state_size + 1) * 5, neurons, batch_first=True)
        self.fc1 = nn.Linear(neurons, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
        self.apply(init_uniform)

    def forward(self, x):
        x, h = self.gru(x)
        x = nn.ReLU()(x)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x[:, -1, :]

class CriticNetwork(nn.Module):
    def __init__(self, n_subcritics=100, neurons=64, state_size=6):
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
    def __init__(self, neurons=64, input_size=16):
        super(SimpleSubCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, neurons)
        self.fc2 = nn.Linear(neurons, neurons)
        self.fc3 = nn.Linear(neurons, 1)
        self.apply(init_uniform)

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
    def __init__(self, actor_lr=1e-4, critic_lr=1e-4, min_action=0, max_action=3):
        super(HRP, self).__init__()
        self.an = ActorNetwork(min_action=min_action, max_action=max_action).to(device)
        self.cn = CriticNetwork().to(device)
        self.an_target = ActorNetwork(min_action=min_action, max_action=max_action).to(device)
        self.cn_target = CriticNetwork().to(device)
        self.critic_criterion = nn.MSELoss()
        self.discount_rate = 0.99
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        self.tau = 0.005
        self.max_action = max_action
        self.critic_optimizer = torch.optim.Adam(self.cn.parameters(), lr=self.critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.an.parameters(), lr=self.actor_lr)
        self.hard_update()
     
    def forward(self, x):
        p = self.an(x)
        xc = torch.cat([x, p.view(-1, 2, 100, 1)], dim=-1)
        q = self.cn(xc)
        return p[:, -1], q.view(-1, 1)
    def target_forward(self, x):
        p = self.an_target(x)
        xc = torch.cat([x, p.view(-1, 2, 100, 1)], dim=-1)
        q = self.cn_target(xc)
        return p[:, -1], q.view(-1, 1)
    def critic_loss(self, batch):
        q = self.cn.forward(torch.cat([batch['state'], batch['action'].view(-1, 2, 100, 1)], dim=-1)).squeeze()
        with torch.no_grad():
            p_next, q_next = self.target_forward(batch['next_state'])
            y = batch['reward'].squeeze() + self.discount_rate * q_next.squeeze() * batch['terminal'].squeeze()
        distance = (y - q).detach().cpu().numpy().mean()
        return self.critic_criterion(y,q), distance

    def soft_update(self):
        for target_param, param in zip(self.an_target.parameters(), self.an.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.cn_target.parameters(), self.cn.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    def hard_update(self):
        for target_param, param in zip(self.an_target.parameters(), self.an.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.cn_target.parameters(), self.cn.parameters()):
            target_param.data.copy_(param.data)

    def critic_step(self, batch):
        self.critic_optimizer.zero_grad()
        batch_loss, _ = self.critic_loss(batch)
        batch_loss.backward()
        self.critic_optimizer.step()
        return batch_loss.detach().cpu().numpy()

    def actor_gradient(self, batch):
        p = self.an(batch['state'])
        xc = torch.cat([batch['state'], p.view(-1, 2, 100, 1)], dim=-1)
        q = self.cn(xc)
        return -1*torch.mean(q.squeeze())
        
    def actor_step(self, batch):
        grad = self.actor_gradient(batch)
        self.actor_optimizer.zero_grad()
        grad.backward()
        torch.nn.utils.clip_grad_norm_(self.an.parameters(), 1.0)
        self.actor_optimizer.step()
        return grad.detach().cpu().numpy()

    def train_step(self, batch):
        batch_loss = self.critic_step(batch)
        q_val = self.actor_step(batch)
        self.soft_update()
        return batch_loss, q_val

class RandomPricing:
    def __init__(self, min_action=0, max_action=3):
        self.min_action = min_action
        self.max_action = max_action
    def an_target(self, state, t=0):
        batch, t, n_zones, state_size = state.size()
        return torch.FloatTensor(batch, t, n_zones).uniform_(self.min_action, self.max_action)


class Agent:

    def __init__(self, name, model, buffer_length, noise_scale=2.0, batch_size=64):
        self.name = name
        self.model = model
        self.experience_buffer = deque(maxlen=buffer_length)
        self.noise_scale = noise_scale
        self.noise_decay = 0.999
        self.batch_size = batch_size
        self.history = {
            'rewards': [],
            'dqn_loss': [],
            'batch_loss': [],
            'distance': []
        }

    def store_transition(self, transition):
        self.experience_buffer.append(transition)

    def sample_minibatch(self):
        batch = random.sample(self.experience_buffer, self.batch_size)
        return [[sample[i] for sample in batch] for i in range(5)]

    def train_minibatch(self, batch_size=32):
        state, action, reward, next_state, terminal = self.sample_minibatch()
        x = {'state': torch.from_numpy(np.concatenate(state)).to(device),
            'action': torch.from_numpy(np.concatenate(action)).to(device), 
            'reward': torch.from_numpy(np.array(reward)).view(-1, 1).to(device),
            'next_state': torch.from_numpy(np.concatenate(next_state)).to(device),
            'terminal': torch.from_numpy(np.array(terminal).astype(np.float32)).to(device)}
        self.model.train()
        self.model.train_step(x)
        self.model.eval()
        return self.model.critic_loss(x)

    def get_action(self, state):
        self.model.an.eval()
        a = self.model.an(state).detach().cpu().numpy()
        self.model.an.train()
        return a

    def get_prices(self, state, t=0):
        state = torch.from_numpy(state.astype(np.float32)).to(device)
        return self.model.an_target(state).detach().cpu().numpy()[:, -1, :]

    def act(self, environment, episode=0):
        state  = environment.get_state()
        action = self.get_action(torch.from_numpy(state).to(device))
        noise = (self.noise_scale * (0.99 ** (episode + 1))) * np.random.normal(size=action.shape)
        print(action)
        action = (action + noise).astype(np.float32)
        action = np.clip(action, 0, self.model.max_action)
        next_state, reward = environment.perform_action(action[:, -1].reshape(10, 10))
        terminal = float(environment.terminal_state)
        self.store_transition((state, action, reward, next_state, terminal))
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
                episode_rewards.append((self.model.discount_rate**t)*reward)
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
            t_bar = trange(environment.timesteps, desc='Episode {}/{}'.format(e, episodes), leave=True)
            for t in t_bar:
                reward = self.act(environment, episode=e)
                episode_rewards.append((self.model.discount_rate**t)*reward)         
                batch_loss, distance = self.train_minibatch()
                t_bar.set_description('Episode {}/{}. Loss = {:.2f}. Reward = {:.2f}'.format(e, episodes, batch_loss.detach().cpu().numpy(), reward))
                self.history['distance'].append(distance)
                self.history['batch_loss'].append(batch_loss.detach().cpu().numpy())
            self.history['rewards'].append(np.sum(episode_rewards))
            self.history['dqn_loss'].append(self.get_q_loss())
        print('Finishing Training: {:.2f}s'.format(time.time() - training_start))

    def get_q_loss(self):
        state, action, reward, next_state, terminal = [[sample[i] for sample in self.experience_buffer] for i in range(5)]
        return self.model.critic_loss({'state': torch.from_numpy(np.concatenate(state)).to(device),
                                    'action': torch.from_numpy(np.concatenate(action)).to(device), 
                                    'reward': torch.from_numpy(np.array(reward)).view(-1, 1).to(device),
                                    'next_state': torch.from_numpy(np.concatenate(next_state)).to(device),
                                    'terminal': torch.from_numpy(np.array(terminal).astype(np.float32)).to(device)})[0].detach().cpu().numpy()
    def save_agent(self, name='test-model'):
        # pass model to cpu to asure loading
        self.model.to(torch.device('cpu'))
        with open('weights/' + name + '.pickle', 'wb') as file:
            pickle.dump(self, file)
    @classmethod
    def load_agent(self, name='test-model'):
        with open('weights/' + name + '.pickle', 'rb') as file:
            agent = pickle.load(file)
            agent.model.to(device)
        return agent

