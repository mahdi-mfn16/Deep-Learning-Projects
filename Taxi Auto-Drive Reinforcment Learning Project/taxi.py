#reinforcement learing with Q-learning
#gym taxi auto-run environment 

import gym
import numpy as np
import random
from IPython.display import clear_output
from time import sleep

env = gym.make('Taxi-v3').env
env.render()



# learning Q_table with eploration on environment

q_table = np.zeros([env.observation_space.n , env.action_space.n])
epsilon = 0.1
alpha = 0.1
landa = 0.6

for i in range(10000):    

  penalties , epochs = 0 , 0
  done = False
  state = env.reset()
  while not done:
    if random.uniform(0,1) < epsilon:
      action = env.action_space.sample()
    else:
      action = np.argmax(q_table[state])

    old_q = q_table[state , action]
    new_state , reward , done , _ = env.step(action)
    next_max = np.max(q_table[new_state])
    new_q = old_q * (1 - alpha) + alpha * (reward + landa * next_max)
    q_table[state , action] = new_q
    if reward == -10:
      penalties += 1
    state = new_state
    epochs += 1

  if i % 10 == 0:
    print('Episode: ' , i)
    print('Epochs: ' , epochs)
    print('Penalties: ' , penalties , '\n')

	
# test of model	

penalties = 0
state = env.reset()
env.render()
sleep(5)
done = False
while not done:
  clear_output(wait = True)
  action = np.argmax(q_table[state])
  new_state , reward , done , _ = env.step(action)
  env.s = new_state
  env.render()
  state = new_state
  if reward == -10:
    penalties +=1
  print('State: ' , new_state)
  print('Reward: ' , reward)
  print('Penalties: ' , penalties)
  sleep(0.5)