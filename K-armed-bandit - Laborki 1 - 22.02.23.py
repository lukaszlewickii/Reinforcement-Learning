import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
# %matplotlib inline

class Bandit:
    def __init__(self, q_star, stddev=2):
        self.stddev = stddev
        self.q_star = q_star

    def execute(self, arm):
        return self.q_star[arm] + np.random.normal(scale=self.stddev)

bandit = Bandit([4, 5, 3, 3, 1, 2, 0, 1, 5, 1])
print(bandit.q_star)
bandit.execute(3)

class RandomAgent:
    def __init__(self, bandit):
        self.q_est = [0] * len(bandit.q_star)
        self.n = [0] * len(bandit.q_star)
        
    def act(self, bandit):
        #Wybierz akcję (losowo)
        arm = np.random.randint(len(self.q_est))
        reward = bandit.execute(arm)
        self.n[arm] +=1
        
        #Zaktualizuj q_est
        self.q_est += (reward - self.q_est[arm]) / self.n[arm]
        
        #Zwróć nagrodę
        return reward
    
agent = RandomAgent(bandit)
rewards = []
for step in range(1000):
    reward = agent.act(bandit)
    rewards.append(reward)