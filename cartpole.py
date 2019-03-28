import gym
#import matplotlib.pyplot as ply
import numpy as np
from policy_gradient import PolicyGradient
env = gym.make('CartPole-v0')
env = env.unwrapped

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)

RENDER_ENV = False
EPISODES = 500
rewards = []
RENDER_REWARD_MIN = 100000

if __name__ == "__main__":

	#load checkpoint
	load_path = "output/weights/CartPole-v0.ckpt"
	save_path = "output/weights/CartPole-v0-temp.ckpt"

	PG = PolicyGradient(
		n_x = env.observation_space.shape[0],
		n_y = env.action_space.n,
		learning_rate = 0.01,
		reward_decay = 0.95,
		load_path = load_path,
		save_path = save_path
		)

	for episode in range(EPISODES):

		observation = env.reset()
		episode_reward = 0

		while True:
			if RENDER_ENV: env.render()

			action = PG.choose_action(observation)

			observation_, reward, done, info = env.step(action)

			PG.store_transition(observation, action, reward)

			#RENDER_ENV = False

			if done:
				episode_rewards_sum = sum(PG.episode_rewards)
				rewards.append(episode_rewards_sum)
				max_reward_so_far = np.amax(rewards)

				print("-------")
				print("Episode: ", episode)
				print("Reward: ", episode_rewards_sum)
				print("Max reward so far: ", max_reward_so_far)

				discounted_episode_rewards_norm = PG.learn()

				#Render if reward > min rewards
				if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True
				if max_reward_so_far < RENDER_REWARD_MIN: RENDER_ENV = False


				break

			observation = observation_

