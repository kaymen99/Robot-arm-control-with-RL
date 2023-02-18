import numpy as np

# Perform HER memory augmentation
def her_augmentation(agent, obs_array, actions, new_obs_array):
    # Hyperparameter for future goals sampling
    k = 4

    # Augment the replay buffer
    size = len(actions)
    for index in range(size):
        for _ in range(k):
            future = np.random.randint(index, size)
            _, future_achgoal, _ = new_obs_array[future].values()

            obs, _, _ = obs_array[future].values()
            state = np.concatenate((obs, future_achgoal, future_achgoal))

            new_obs, _, _ = new_obs_array[future].values()
            next_state = np.concatenate((new_obs, future_achgoal, future_achgoal))

            action = actions[future]
            reward = agent.env.compute_reward(future_achgoal, future_achgoal, 1.0)

            # Store augmented experience in buffer
            agent.remember(state, action, reward, next_state, True)