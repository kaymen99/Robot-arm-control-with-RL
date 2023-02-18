import tensorflow as tf
import tensorflow.keras as keras
from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks import ActorNetwork, CriticNetwork

## Actor-critic networks parameters :

# actor learning rate
alpha = 0.001

# critic learning rate
beta = 0.002

## DDPG algorithms paramters

# discount factor
gamma = 0.99

# target netwroks soft update factor 
tau = 0.005

# replay buffer max memory size
max_size = 10**6

# exploration noise factor 
noise_factor = 0.1

# training batch size 
batch_size = 64

## DDPG agent class 
class DDPGAgent:
    def __init__(self, env, input_dims):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_factor = noise_factor

        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)

        self._initialize_networks(self.n_actions)
        self.update_parameters(tau=1)

    # Choose action based on actor network
    # Add exploration noise if in traning mode
    def choose_action(self, state, evaluate=False):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], mean=0, stddev=self.noise_factor)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0]
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    # Main DDPG algorithms learning process
    def learn(self):
          if self.memory.counter < self.batch_size:
              return

          # Sample batch size of experiences from replay buffer
          states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
          states = tf.convert_to_tensor(states, dtype=tf.float32)
          actions = tf.convert_to_tensor(actions, dtype=tf.float32)
          rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
          new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

          # Calculate critic network loss
          with tf.GradientTape() as tape:
              target_actions = self.target_actor(new_states)
              new_critic_value = tf.squeeze(self.target_critic(new_states, target_actions), 1)
              critic_value = tf.squeeze(self.critic(states, actions), 1)
              target = rewards + self.gamma * new_critic_value * (1 - dones)
              critic_loss = tf.keras.losses.MSE(target, critic_value)

          # Apply gradient decente with the calculated critic loss
          critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
          self.critic.optimizer.apply_gradients(zip(
              critic_network_gradient, self.critic.trainable_variables 
          ))

          # Calculate actor network loss
          with tf.GradientTape() as tape:
              new_actions = self.actor(states)
              actor_loss = - self.critic(states, new_actions)
              actor_loss = tf.math.reduce_mean(actor_loss)
          
          # Apply gradient decente with the calculated actor loss
          actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
          self.actor.optimizer.apply_gradients(zip(
                  actor_network_gradient, self.actor.trainable_variables 
              ))
          
          # Update actor/critic target networks
          self.update_parameters()

    # Update actor/critic target networks parameters with soft update rule
    def update_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(tau * weight + (1 - tau) * targets[i])
        self.target_critic.set_weights(weights)

    def save_models(self):
        print("---- saving models ----")
        self.actor.save_weights(self.actor.checkpoints_file)
        self.critic.save_weights(self.critic.checkpoints_file)
        self.target_actor.save_weights(self.target_actor.checkpoints_file)
        self.target_critic.save_weights(self.target_critic.checkpoints_file)

    def load_models(self):
        print("---- loading models ----")
        self.actor.load_weights(self.actor.checkpoints_file)
        self.critic.load_weights(self.critic.checkpoints_file)
        self.target_actor.load_weights(self.target_actor.checkpoints_file)
        self.target_critic.load_weights(self.target_critic.checkpoints_file)

    def _initialize_networks(self, n_actions):
        model = "ddpg"
        self.actor = ActorNetwork(n_actions, name="actor", model=model)
        self.critic = CriticNetwork(name="critic", model=model)
        self.target_actor = ActorNetwork(n_actions, name="target_actor", model=model)
        self.target_critic = CriticNetwork(name="target_critic", model=model)

        self.actor.compile(keras.optimizers.Adam(learning_rate=alpha))
        self.critic.compile(keras.optimizers.Adam(learning_rate=beta))
        self.target_actor.compile(keras.optimizers.Adam(learning_rate=alpha))
        self.target_critic.compile(keras.optimizers.Adam(learning_rate=beta))