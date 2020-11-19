from  tensorflow.keras.models import load_model
import os

load_dir = os.getcwd()
model_name = 'keras_dqn_trained_model_3000.h5'
model_path = os.path.join(load_dir, model_name)
model = load_model(model_path)

import gym
from gym.wrappers import Monitor
import numpy as np

env = gym.make('CartPole-v1')
num_state = env.observation_space.shape[0]
state = env.reset()
done = False

mem = []

while not done:
    env.render()
    state = np.array(state).reshape(1, num_state)
    q_value = model.predict(state)
    mem.append(q_value[0])
    action = np.argmax(q_value[0])
    state, reward, done, info = env.step(action)

#file_infix = env.file_infix
env.close()

#show_video(file_infix)