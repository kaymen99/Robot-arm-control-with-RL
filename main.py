import gymnasium as gym
import panda_gym
from numpngw import write_apng
from IPython.display import Image
from agents.ddpg import DDPGAgent

env = gym.make("PandaReach-v3", render_mode="rgb_array")
obs_shape = env.observation_space['observation'].shape[0] + \
                env.observation_space['achieved_goal'].shape[0] + \
                env.observation_space['desired_goal'].shape[0]

# Choose your trained agent : DDPG or TD3
agent = DDPGAgent(env=env, input_dims=obs_shape)
# load pre-trained networks weights
agent.load_models()

observation, info = env.reset()

# Stores frames of robot arm moving in Reacher env
images = [env.render()]

done = False
truncated = False
for i in range(200):
    curr_obs, curr_achgoal, curr_desgoal = observation.values()
    state = np.concatenate((curr_obs, curr_achgoal, curr_desgoal))

    # Choose an action using pre-trainded RL agent
    action = agent.choose_action(state)

    # Excute the choosen action in the environement
    new_observation, reward, done, truncated, _ = env.step(np.array(action))
    images.append(env.render())
    observation = new_observation

    if done or truncated:
        observation, info = env.reset()
        images.append(env.render())

env.close()

# save frames : real-time rendering = 40 ms between frames
write_apng("anim.png", images, delay=60)
# show movements
Image(filename="anim.png")
