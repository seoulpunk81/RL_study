import gym
import tensorflow as tf
import numpy as np

# CartPole 환경 구성
env = gym.make('CartPole-v0')

# 뉴럴 네트워크 모델 만들기
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, input_shape=(4,), activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

# 첫번째 관찰
observation = env.reset()

# 뉴럴 네트워크의 선택
predict = model.predict(observation.reshape(1, 4))
action = np.argmax(predict)

print(observation)
print(predict)
print(action)