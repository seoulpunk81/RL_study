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

score = []

# 100회의 에피소드
for i in range(100):
  observation = env.reset()

  # 200개의 시간 스텝
  for t in range(200):

      # 뉴럴 네트워크의 선택
      predict = model.predict(observation.reshape(1, 4))
      action = np.argmax(predict)

      observation, reward, done, info = env.step(action)

      if done:
          score.append(t + 1)
          break

env.close()
print(score)