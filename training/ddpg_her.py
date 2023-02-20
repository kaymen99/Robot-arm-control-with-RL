import numpy as np
import gymnasium as gym
import panda_gym
from agents.ddpg import DDPGAgent
from utils.HER import her_augmentation


if __name__ == "__main__":

    n_games = 1500
    opt_steps = 64
    best_score = 0
    score_history = []
    avg_score_history = []
    
    env = gym.make('PandaReach-v3')
    obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

    agent = DDPGAgent(env=env, input_dims=obs_shape)

    for i in range(n_games):
        done = False
        truncated = False
        score = 0
        step = 0

        obs_array = []
        actions_array = []
        new_obs_array = []

        observation, info = env.reset()

        while not (done or truncated):
            curr_obs, curr_achgoal, curr_desgoal = observation.values()
            state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

            # Choose an action
            action = agent.choose_action(state, False)

            # Excute the choosen action in the environement
            new_observation, reward, done, truncated, _ = env.step(np.array(action))
            next_obs, next_achgoal, next_desgoal = new_observation.values()
            new_state = np.concatenate((next_obs, next_achgoal, next_desgoal))

            # Store experience in the replay buffer
            agent.remember(state, action, reward, new_state, done)
        
            obs_array.append(observation)
            actions_array.append(action)
            new_obs_array.append(new_observation)

            observation = new_observation
            score += reward
            step += 1
        
        # Augmente replay buffer with HER
        her_augmentation(agent, obs_array, actions_array, new_obs_array)

        # train the agent in multiple optimization steps
        for _ in range(opt_steps):
          agent.learn()
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        avg_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
        
        print(f"Episode {i} steps {step} score {score:.1f} avg score {avg_score:.1f}")
