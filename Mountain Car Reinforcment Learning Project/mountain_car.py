#reinforcement learing with Q-learning with sub-classing model
#gym mountain-car environment 

import numpy as np
import gym
import random
from time import sleep
from IPython.display import clear_output , display
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque

env = gym.make('MountainCar-v0')

class DDQN:
  def __init__(self , env):
      self.env = env
      self.gamma = 0.9
      self.epsilon = 1
      self.tau = 0.125
      self.epsilon_decay = 0.995
      self.alpha = 0.2
      self.epsilon_min = 0.01
      self.memory = deque(maxlen = 2000)
      self.batch_size = 32
      self.model = self.create_model()
      self.target_model = self.create_model()

  def create_model(self):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(16 , input_shape = (self.env.observation_space.shape[0],) , activation = 'relu'))
    model.add(tf.keras.layers.Dense(32 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(16 , activation = 'relu'))
    model.add(tf.keras.layers.Dense(self.env.action_space.n))
    model.compile(
        loss = 'mean_squared_error',
        optimizer = tf.keras.optimizers.Adam(0.001)
    )
    return model

  def act(self , state):
    self.epsilon *= self.epsilon_decay
    self.epsilon = max(self.epsilon , self.epsilon_min)
    if np.random.uniform() < self.epsilon:
      return self.env.action_space.sample()
    else:
      return np.argmax(self.model.predict(state)[0])

  def replay(self):
    if len(self.memory) < self.batch_size:
      return
    else:
      samples = random.sample(self.memory , self.batch_size)
      for sample in samples:
        state , action , new_state , reward , done = sample
        target = self.target_model.predict(state)
        if done:
          target[0][action] = reward
        else:
          q_new_max = max(self.target_model.predict(new_state)[0])
          target[0][action] = target[0][action] * self.alpha + (reward + q_new_max * self.gamma) * (1 - self.alpha)
        self.model.fit(state , target , epochs = 1)

  def create_memory(self , state , action , new_state , reward , done):
    self.memory.append([state , action , new_state , reward , done])

  def target_train(self):
    weights = self.model.get_weights()
    target_weights = self.target_model.get_weights()
    for i in range(len(weights)):
      target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
    self.target_model.set_weights(target_weights)

  def save_model(self , file_name):
    self.model.save(file_name)

ddqn = DDQN(env = env)

trails = 1000
trail_size = 500
for trail in range(trails):
  print('#trail:' , trail)
  curr_state = env.reset().reshape(1,2)
  for step in range(trail_size):
    clear_output()
    print('#step:' , step)
    action = ddqn.act(curr_state)
    new_state , reward , done , _ = env.step(action)
    new_state = new_state.reshape(1,2)
    ddqn.create_memory(curr_state, action, new_state, reward, done)
    ddqn.replay()
    ddqn.target_train()
    curr_state = new_state
    if done:
      break
  if step < 199:
    print('success')
    ddqn.save_model('model')
    break
  else:
    print('failed')