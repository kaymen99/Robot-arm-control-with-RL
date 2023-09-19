import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from replay_memory.ReplayBuffer import ReplayBuffer
from utils.networks import ActorNetwork, CriticNetwork

## Actor-critic networks parameters :
# actor learning rate
alpha = 0.001

# critic learning rate
beta = 0.002

## TD3 algorithms paramters
# discount factor
gamma = 0.99

# target netwroks soft update factor
tau = 0.05

# replay buffer max memory size
max_size = 10**6

# exploration noise factor
noise_factor = 0.1

# training batch size
batch_size = 256

class TD3Agent:
    def __init__(self, env, input_dims, update_actor_interval=2, warmup=500):
        # setup hyperparameters values
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.time_step = 0
        self.warmup = warmup
        self.learn_step_counter = 0
        self.update_actor_interval = update_actor_interval
        self.noise_factor = noise_factor

        # setup environment
        self.env = env
        self.n_actions = env.action_space.shape[0]
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]

        # setup replay buffer memory
        self.memory = ReplayBuffer(max_size, input_dims, self.n_actions)

        # initialize actor and critic netwprks
        self._initialize_networks(self.n_actions)
        self.update_parameters(tau=1)

    def choose_action(self, observation):
        """
        Choose an action for the agent.

        If the time step is less than the warmup period, actions are selected randomly
        from a normal distribution. This encourages exploration in the early stages,
        with 'warmup' being a tunable hyperparameter.

        After the time step is greater than or equal to the warmup period, actions are
        determined using the actor network, with some noise added to the network's output
        to promote exploration.

        Args:
            observation : The current observation/state of the environment.

        Returns:
            action: The chosen action.
        """
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise_factor, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0]
        mu_star = mu + np.random.normal(scale=self.noise_factor)
        mu_star = tf.clip_by_value(mu_star, self.min_action, self.max_action)
        self.time_step += 1

        return mu_star

    def remember(self, state, action, reward, new_state, done):
        """
        Interface function between agent and buffer, used to store transitions
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """
        Main agent learning function implementing the TD3 algorithm.

        This function trains the agent using the following steps:
        1. Sample a random batch of old experiences from memory.
        2. Apply gradient descent on the two critic networks.
        3. Apply gradient descent on the actor network in a delayed
        manner: actor is updated once for every two updates of critic networks.

        Returns:
            None
        """

        # Check if there are enough experiences in memory to begin training
        if self.memory.counter < self.batch_size:
            return

        # Step 1: Sample a random batch of old experiences from memory
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        new_states = tf.convert_to_tensor(new_states, dtype=tf.float32)

        # Step 2: Apply gradient descent on the two critic networks
        with tf.GradientTape(persistent=True) as tape:
            # Calculate target actions with some noise
            target_actions = self.target_actor(new_states)
            target_actions += tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            # Compute Q-values and target Q-values for both critic networks
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)
            q1_new = tf.squeeze(self.target_critic_1(new_states, target_actions), 1)
            q2_new = tf.squeeze(self.target_critic_2(new_states, target_actions), 1)
            target = rewards + self.gamma * tf.math.minimum(q1_new, q2_new) * (1 - dones)

            # Compute critic losses
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        # Compute critic gradients and apply gradient descent
        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        # Step 3: Update the actor network only once for every two updates of critic networks
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_actor_interval != 0:
            return

        # Apply gradient descent on the actor network
        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        # Compute actor gradients and apply gradient descent
        actor_gardient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gardient, self.actor.trainable_variables))

        # Update actor/critic target networks weights with soft update rule
        self.update_parameters()

    def update_parameters(self, tau=None):
        """
        Update the weights of the target actor and both target critic networks using a soft update rule.

        The formula used for the soft update is as follows:
            new_weight = tau * old_weight + (1 - tau) * old_target_weight

        Args:
            tau (float, optional): The interpolation parameter for the soft update.
                If not provided, the default tau value from the class attributes is used.

        Returns:
            None
        """
        if tau is None:
            tau = self.tau

        # Update the weights of the target actor
        self._update_target_network(self.target_actor, self.actor, tau)

        # Update the weights of the first target critic network
        self._update_target_network(self.target_critic_1, self.critic_1, tau)

        # Update the weights of the second target critic network
        self._update_target_network(self.target_critic_2, self.critic_2, tau)

    def save_models(self):
        print("---- saving models ----")
        self.actor.save_weights(self.actor.checkpoints_file)
        self.critic_1.save_weights(self.critic_1.checkpoints_file)
        self.critic_2.save_weights(self.critic_2.checkpoints_file)
        self.target_actor.save_weights(self.target_actor.checkpoints_file)
        self.target_critic_1.save_weights(self.target_critic_1.checkpoints_file)
        self.target_critic_2.save_weights(self.target_critic_2.checkpoints_file)

    def load_models(self):
        print("---- loading models ----")
        self.actor.load_weights(self.actor.checkpoints_file)
        self.critic_1.load_weights(self.critic_1.checkpoints_file)
        self.critic_2.load_weights(self.critic_2.checkpoints_file)
        self.target_actor.load_weights(self.target_actor.checkpoints_file)
        self.target_critic_1.load_weights(self.target_critic_1.checkpoints_file)
        self.target_critic_2.load_weights(self.target_critic_2.checkpoints_file)

    def _initialize_networks(self, n_actions):
        model = "TD3"
        self.actor = ActorNetwork(n_actions, name="actor", model=model)
        self.critic_1 = CriticNetwork(name="critic_1", model=model)
        self.critic_2 = CriticNetwork(name="critic_2", model=model)

        self.target_actor = ActorNetwork(n_actions, name="target_actor", model=model)
        self.target_critic_1 = CriticNetwork(name="target_critic_1", model=model)
        self.target_critic_2 = CriticNetwork(name="target_critic_2", model=model)

        self.actor.compile(keras.optimizers.Adam(learning_rate=alpha), loss="mean")
        self.critic_1.compile(keras.optimizers.Adam(learning_rate=beta), loss="mean_squared_error")
        self.critic_2.compile(keras.optimizers.Adam(learning_rate=beta), loss="mean_squared_error")

        self.target_actor.compile(keras.optimizers.Adam(learning_rate=alpha), loss="mean")
        self.target_critic_1.compile(keras.optimizers.Adam(learning_rate=beta), loss="mean_squared_error")
        self.target_critic_2.compile(keras.optimizers.Adam(learning_rate=beta), loss="mean_squared_error")

    def _update_target_network(self, target_network, source_network, tau):
        """
        Update the weights of a target neural network using a soft update rule.

        Args:
            target_network (tf.keras.Model): The target neural network whose weights need to be updated.
            source_network (tf.keras.Model): The source neural network from which weights are copied.
            tau (float): The interpolation parameter for the soft update.

        Returns:
            None
        """
        weights = []
        target_weights = target_network.weights
        for i, weight in enumerate(source_network.weights):
            weights.append(tau * weight + (1 - tau) * target_weights[i])
        target_network.set_weights(weights)
