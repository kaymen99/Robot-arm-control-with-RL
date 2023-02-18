import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.size = max_size
        self.counter = 0
        self.states = np.zeros((max_size, input_shape))
        self.actions = np.zeros((max_size, n_actions))
        self.rewards = np.zeros(max_size)
        self.new_states = np.zeros((max_size, input_shape))
        self.terminal_state = np.zeros(max_size)
    
    def store_transition(self, state, action, reward, new_state, done):
        index = self.counter % self.size

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.new_states[index] = new_state
        self.terminal_state[index] = done

        self.counter += 1

    def sample(self, batch_size):
        max_memory = min(self.size, self.counter)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = self.states[batch]
        actions = self.actions[batch]
        rewards = self.rewards[batch]
        new_states = self.new_states[batch]
        dones = self.terminal_state[batch]

        return states, actions, rewards, new_states, dones