import gym

env = gym.make('CartPole-v0')
observation = env.reset()
action = env.action_space.sample()

print(action)