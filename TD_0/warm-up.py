import gym
import gym_walking
import numpy as np
import time

env = gym.make('Walking5-v0')
def pi(x): return np.random.randint(2)


state, info = env.reset()
env.render()
terminal = False

while not terminal:
    action = pi(state)
    next_state, reward, terminal, truncated, info = env.step(action)
    env.render()
    print(f'State: {state}, action: {action}, next state: {next_state}, reward: {reward}, terminal: {terminal}')
    state = next_state

