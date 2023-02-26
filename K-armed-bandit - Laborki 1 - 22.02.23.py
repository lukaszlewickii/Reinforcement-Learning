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
        # Wybierz akcję (losowo)
        arm = np.random.randint(len(self.q_est))
        reward = bandit.execute(arm)
        self.n[arm] += 1

        # Zaktualizuj q_est
        self.q_est[arm] += (reward - self.q_est[arm]) / self.n[arm]

        # Zwróć nagrodę
        return reward


agent = RandomAgent(bandit)
rewards = []
for step in range(1000):
    reward = agent.act(bandit)
    rewards.append(reward)


def plot_rewards(rewards):
    
    """
    Plots the rewards
    """
    f = plt.figure()
    plt.plot(rewards)
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.show()


plot_rewards(rewards)


def display_agent(agent, bandit):
    
    """
    Displays the agent data, namely:
    q_star - real expected rewards from the bandit
    q_est  - expected rewards extimated by the agent
    n      - how many times the agent selected this particular arm
    """
    df = pd.DataFrame()
    df['q_star'] = bandit.q_star
    df['q_est'] = agent.q_est
    df['n'] = agent.n
    return df


display_agent(agent, bandit)


### EPSILON-GREEDY AGENT ###

class EpsilonGreedyAgent:
    def __init__(self, bandit, epsilon=0.1):
        self.q_est = [0] * len(bandit.q_star)
        self.n = [0] * len(bandit.q_star)
        self.epsilon = epsilon

    def act(self, bandit):
        # Wybierz ramię
        if np.random.random() > self.epsilon:
            arm = np.argmax(self.q_est)
        else:
            arm = np.random.randint(len(self.q_est))

        reward = bandit.execute(arm)
        self.n[arm] += 1

        self.q_est[arm] += (reward - self.q_est[arm]) / self.n[arm]

        return reward

"""
total_rewards = []
for loop in tqdm(range(2000)):
    agent = EpsilonGreedyAgent(bandit)
    rewards = []
    for step in range(1000):
        reward = agent.act(bandit)
        rewards.append(reward)
    total_rewards.append(rewards)

total_rewards = np.asarray(total_rewards)
rewards = total_rewards.mean(axis=0)
print(rewards.shape)
"""

### Task 1 - Optimistic agent

class OptimisticAgent:
    def __init__(self, bandit, initial_value = 10):
        """
        Arguments:
            bandit - bandit that the agent will operate on (used only to set initial values of q_est and n)
            initial_value - initial value for estimated q_star. It should be high enough to make the algorithm work
        """
        self.q_est = [0] * len(bandit.q_star)
        self.n = [0] * len(bandit.q_star)
    
    def act(self, bandit):    
        """
        Performs an action (selects an arm greedily based on the estimated q_star values ) 
        and updates corespondingn q_est and n values
        Arguments:
            bandit - bandit that the agent is operated on
        Returns:
            reward - reward from the bandit (a result of the performed action)
        """
        arm = np.random.randint(len(self.q_est))
        reward = bandit.execute(arm)
        self.n[arm] += 1
        
        # Q_n+1 = Qn + (1 / n) * (R - Q_n)
        self.q_est = [0] * len(bandit.q_star)
        
        return reward
    
total_rewards = []
for loop in tqdm(range (2000)):
    agent = OptimisticAgent(bandit)
    rewards = []
    for step in range (1000):
        reward = agent.act(bandit)
        rewards.append(reward)
    
    total_rewards.append(rewards)
total_rewards = np.asarray(total_rewards)
rewards = total_rewards.mean(axis=0)

#df_rewards['optimistic'] = rewards # Let's store the results for comparison
plot_rewards(rewards)
display_agent(agent, bandit)

class UCBAgent:
    def __init__(self, bandit, c = 3):
        self.q_est =  [0] * len(bandit.q_star)
        self.n =  [0] * len(bandit.q_star)
        self.c = c
        
    def act(self, bandit): 
        # Calculate Q optimistic here
        # Enter your code here (probably you need more than one line of code)
        
        arm = np.random.randint(len(self.q_est))
        reward = bandit.execute(arm)
        self.n[arm] += 1
        
        # Q_n+1 = Qn + (1 / n) * (R - Q_n)
        self.q_est[arm] += self.c * np.sqrt((np.log(self.n))/) # Enter your code here
        
        return reward