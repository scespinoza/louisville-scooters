from itertools import product
from collections import deque
import random
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt


gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

class SubActor(layers.Layer):
    def __init__(self, n_neurons=8, n_hidden=2, name='SubActor'):
        super(SubActor, self).__init__(name=name)
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden
        self.layers = [layers.Dense(self.n_neurons, activation='sigmoid',
                        kernel_initializer='he_normal', kernel_regularizer='l2') 
                      for _ in range(n_hidden)]
        self.output_layer = layers.Dense(1, activation='relu')
    
    def call(self, x):
        for n in tf.range(len(self.layers)):
            x = self.layers[n](x)
        return self.output_layer(x)

class ActorNetwork(models.Model):

    def __init__(self, n_regions, name='ActorNetwork'):
        super(ActorNetwork, self).__init__(name=name)
        self.n_regions = n_regions
        for i in range(n_regions):
            setattr(self, "gru_%i" % i, layers.GRU(16))
            setattr(self, "subactor_%i" % i, SubActor())
    
    def call(self, x):
        batch_size, y_regions, x_regions, t, state_size = x.shape
        prices = []
        for n in tf.range(100):
            i, j = n // 10, n % 10
            x_p = getattr(self, "gru_{}".format(n))(x[:, i, j, :])
            price = getattr(self, "subactor_{}".format(n))(x_p)
            prices.append(price)
        return tf.reshape(prices, shape=(batch_size, y_regions, x_regions))


class LocalizedModule(layers.Layer):

    def __init__(self, n_neurons=8, n_hidden=2, name='LocalizedModule'):
        super(LocalizedModule, self).__init__(name=name)
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden
        self.layers = [layers.Dense(self.n_neurons, activation='sigmoid',
                                    kernel_initializer='zeros')
                        for _ in range(self.n_hidden)]
        self.output_layer = layers.Dense(1, activation='linear')

    def call(self, x):
        for n in tf.range(len(self.layers)):
            x = self.layers[n](x)
        return self.output_layer(x)

class SubCritic(layers.Layer):
    def __init__(self, n_neurons=8, n_hidden=2, name='SubCritic'):
        super(SubCritic, self).__init__(name=name)
        self.n_neurons = n_neurons
        self.n_hidden = n_hidden
        self.layers = [layers.Dense(self.n_neurons, activation='sigmoid',
                        kernel_initializer='zeros') 
                      for _ in range(n_hidden)]
        self.output_layer = layers.Dense(1, activation='linear')

    def call(self, x):
        for n in tf.range(len(self.layers)):
            x = self.layers[n](x)
        return self.output_layer(x)
        
class CriticNetwork(models.Model):
    def __init__(self, n_regions, name='CriticNetwork'):
        super(CriticNetwork, self).__init__(name=name)
        self.n_regions = n_regions
        self.region_critics = []
        for i in range(n_regions):
            setattr(self, "gru_%i" % i, layers.GRU(16, kernel_initializer='zeros'))
            setattr(self, "loc_module_%i" % i, LocalizedModule(name='LocalizedModule-Region-{}'.format(i)))
            setattr(self, "sub_critic_%i" % i, SubCritic(name='SubCritic-Region-{}'.format(i)))

    def call(self, x):
        x, p = x
        batch_size, y_regions, x_regions, t, state_size = x.shape
        f = []
        q = []
        p = tf.stack([p, p], axis=3)
        for n in tf.range(100):
            i, j = n // 10, n % 10
            i_area = i + 1
            j_area = j + 1
            paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0], [0, 0]])

            x = tf.pad(x, paddings=paddings)
            neighbors = [(i_area - 1, j_area), (i_area, j_area + 1), (i_area + 1, j_area), (i_area, j_area - 1)]

            neighbors_tensor = tf.stack([x[:, i, j] for i, j in neighbors])

            neighbors = tf.reshape(tf.transpose(neighbors_tensor, 
                                    perm=[1, 2, 0, 3]), 
                                    shape=(batch_size, t, state_size * 4))
            x_f = tf.concat([neighbors, x[:, i, j], tf.reshape(p[:, i, j], shape=(batch_size, t, 1))], axis=2)
            x_q = tf.concat([x[:, i, j], tf.reshape(p[:, i, j], shape=(batch_size, t, 1))], axis=2)
            f.append(getattr(self, "loc_module_{}".format(n))(tf.reshape(x_f, shape=(batch_size, t * (5 * state_size + 1) ))))
            q.append(getattr(self, "sub_critic_{}".format(n))(getattr(self, "gru_{}".format(n))(x_q)))
        Q = tf.reduce_sum(tf.stack(f) +  tf.stack(q))
        return Q
        

class HRP(models.Model):
    def __init__(self, regions=(10, 10), name='HRP', target=False, learning_rate=1e-3):
        super(HRP, self).__init__(name=name)
        self.regions = regions
        self.n_regions = regions[0] * regions[1]
        self.actor_network = ActorNetwork(n_regions=100)
        self.critic_network = CriticNetwork(n_regions=100)
        self.optimizer = optimizers.Adam(learning_rate)
        self.discount_rate = 0.99
        self.theta = 0.2
        self.tau = 0.2
        if not target:
            self.target_network = HRP(regions=regions, name=name + '-Target', target=True)

    def call(self, state):
        p = self.actor_network(state)
        q = self.critic_network([state, p])
        return q

    def dqn_loss(self, x):
        state, action, reward, next_state = x
        q_next_state = self.target_network(next_state)
        y = reward + self.discount_rate * q_next_state
        return tf.reduce_mean(tf.square(y - self.critic_network([state, action])))

    
    def critic_gradients(self, batch):
        state, action, reward, next_state = batch
        with tf.GradientTape() as tape:
            loss = self.dqn_loss(batch)
            return tape.gradient(loss, self.critic_network.trainable_variables)

    
    def actor_gradients(self, batch):
        state, _, reward, next_state = batch
        with tf.GradientTape() as tape:
            action = self.actor_network(state)
            action_gradient = self.action_gradient([state, action])
            return tape.gradient(action, self.actor_network.trainable_variables,-action_gradient)
    def action_gradient(self, x):
        state, action = x
        action = tf.constant(action)
        with tf.GradientTape() as tape:
            tape.watch(action)
            q_value = self.critic_network(x)
            return tape.gradient(q_value, action)

    def train_step(self, batch):
        actor_gradients = self.actor_gradients(batch)
        critic_gradients = self.critic_gradients(batch)
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))
        self.update_target()

    def update_target(self):
        target_critic_weights = self.target_network.critic_network.get_weights()
        online_critic_weights = self.critic_network.get_weights()
        target_actor_weights = self.target_network.actor_network.get_weights()
        online_actor_weights = self.actor_network.get_weights()
        
        target_critic_new_weights = [self.tau * online_weights + (1 - self.tau) * target_weights
                                     for online_weights, target_weights in zip(online_critic_weights, target_critic_weights)]
        target_actor_new_weights = [self.tau * online_weights + (1 - self.tau) * target_weights
                                     for online_weights, target_weights in zip(online_actor_weights, target_actor_weights)]
        
        self.target_network.critic_network.set_weights(target_critic_new_weights)
        self.target_network.actor_network.set_weights(target_actor_new_weights)

    def save_target_model(self, name='test_model'):
        self.save_weights('weights/{}.h5'.format(name))
    
    def load_trained(self, name='test_model'):
        self(Environment().get_state())
        self.load_weights('weights/{}.h5'.format(name))


class Agent:

    def __init__(self, name, model, buffer_length, batch_size):
        self.name = name
        self.model = model
        self.batch_size=batch_size
        self.experience_buffer = deque(maxlen=buffer_length)
        self.random_exploration_process = np.random.rand
        self.history = {
            'rewards': [],
            'dqn_loss': [],
            'batch_loss': []
        }

    def store_transition(self, transition):
        self.experience_buffer.append(transition)

    def sample_minibatch(self, batch_size):
        batch = random.sample(self.experience_buffer, batch_size)
        return [[sample[i] for sample in batch] for i in range(4)]

    def train_minibatch(self):
        state, action, reward, next_state = self.sample_minibatch(self.batch_size)
        x = [np.concatenate(state), np.concatenate(action), np.array(reward), np.concatenate(next_state)]
        self.model.train_step(x)
        return self.model.dqn_loss(x)

    def get_action(self, state):
        return self.model.actor_network(state).numpy()

    def act(self, environment):
        state  = environment.get_state()
        action = self.get_action(state)
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
                print('Experience Buffer: {} samples'.format(len(self.experience_buffer)))
                reward = self.act(environment)
                print('Reward: {:.2f}, '.format(reward), end='')  
                episode_rewards.append(self.model.discount_rate*reward)         
                batch_loss = self.train_minibatch()
                print('Batch Loss: {:.2f}'.format(batch_loss))
                self.history['batch_loss'].append(batch_loss)
                plt.figure()
                plt.plot(self.history['batch_loss'])
                plt.savefig('figures/training-loss-{}-{}.png'.format(e, t))
            self.history['rewards'].append(np.sum(episode_rewards))
            self.history['dqn_loss'].append(self.get_q_loss())
        self.model.save_target_model('test_model')
        print('Finishing Training: {:.2f}s'.format(time.time() - training_start))

    def get_q_loss(self):
        state, action, reward, next_state = [[sample[i] for sample in self.experience_buffer] for i in range(4)]
        return self.model.dqn_loss([np.concatenate(state), np.concatenate(action), np.array(reward), np.concatenate(next_state)])
    def save_target_model(self):
        self.model.save_target_model()

    def load_trained(self):
        self.model.load_trained()
        return self


class Environment:
    def __init__(self, name='Environment'):
        self.name = name
        self.timesteps = 24

    def get_state(self):
        return np.random.randint(50, size=(1, 10, 10, 1, 6)).astype(np.float32)

    def perform_action(self, action):
        return np.random.uniform()

if __name__ == '__main__':
    print('Testing model input')
    dummy_data = np.random.randint(0, 50, size=(2, 10, 10, 1, 6)).astype('float32')
    model = HRP()
    agent = Agent(name='test-agent', model=model, buffer_length=100)
    environment = Environment(name='test-environment')
    agent.train(environment, warmup_iterations=1, episodes=1)
    plt.plot(agent.rewards)
    plt.show()