import gym
import numpy as np

#
# Environment
#

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


env = gym.make('CartPole-v1')
state = env.reset()
action = env.action_space.sample()

print('State space: ', env.observation_space)
print('Initial state: ', state)
print('\nAction space: ', env.action_space)
print('Random action: ', action)

# DQN Modeling
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import random
from collections import deque
from tqdm import tqdm
import os

save_dir = os.getcwd()

num_episode = 3000
#num_episode = 7000
memory = deque(maxlen=2000)

# Hyper parameter
epsilon = 0.3
gamma = 0.95
batch_size = 32

num_state = env.observation_space.shape[0]
num_action = env.action_space.n

with tf.device('/gpu:0'):
    
    model = Sequential()
    model.add(Dense(24, input_dim= num_state, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(num_action, activation='linear'))
    model.compile(loss='mse', optimizer="adam")

    # DQN Learning
    for episode in tqdm(range(num_episode)):
        state = env.reset()
        done = False    
        while not done:
            #env.render()
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                q_value = model.predict(state.reshape(1, num_state))
                action = np.argmax(q_value[0])
            next_state, reward, done, info = env.step(action)
            # Memory
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
        
        # Replay
        if len(memory) > batch_size:
            mini_batch = random.sample(memory, batch_size)
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    target = reward + gamma * (np.max(model.predict(next_state.reshape(1, num_state))[0]))
                q_value = model.predict(state.reshape(1, num_state))
                q_value[0][action] = target
                model.fit(state.reshape(1, num_state), q_value, epochs=1, verbose=0)

    env.close()


save_dir = os.getcwd()
model_name = 'keras_dqn_trained_model_3000.h5'

# Save model and weights
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
