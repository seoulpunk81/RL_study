import gym

env = gym.make('CartPole-v0')
observation = env.reset()

# [ 카트 위치, 속도, pole 각도, 회전율 ]
print(observation)