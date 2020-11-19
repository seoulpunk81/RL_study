import gym

env = gym.make('CartPole-v0')
observation = env.reset()
action = env.action_space.sample()
step = env.step(action)

print('First observation:', observation)
print('Action:', action)
print('Step:', step)
# step = ( observation, reward, done, info)