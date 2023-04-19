import numpy as np

class SarsaAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        """ Initialize the agent.
        Args:
            env (gym.Env): environment to learn from.
            alpha (float): learning rate.
            gamma (float): discount factor.
            epsilon (float): epsilon-greedy policy parameter.
        """
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Sets the Q_est to a matrix of zeros with the shape of (nS, nA)
        self.Q_est = np.array([np.zeros(env.action_space.n) for _ in range(env.observation_space.n)])
        
    def get_action(self, state):
        """
        Get an action from Q_est using epsilon-greedy policy. Action is
        optimal with the probability of (1 - epsilon) + epsilon / nA.
        Otherwise, action is selected uniformly at random.
        Args:
            state (int): current state
        Returns:
            action (int): action to take.
        """
        # TODO: Implement the get_action function for SARSA agent.
        # YOUR CODE HERE
        action = 0
        if np.random.uniform(0, 1) < self.epsilon:
            action = self.action_space.sample()
        else:
            action = np.argmax(self.Q[state, :])
        return action
        
    def update(self, state, action, reward, next_state, next_action):
        """ Updates the Q_est using the SARSA update rule.
        Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
        Args:
            state (int): state at time t
            action (int): action at time t
            reward (int, float): received reward
            next_state (int): state at time t+1
            next_action (int): action at time t+1
        """
        # TODO: Implement the update function for SARSA agent.
        # YOUR CODE HERE
        
        predict = self.Q[state, action]
        target = reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - predict)
        pass
    
    def save(self, path = './SARSA_Q_est.npy'):
        """ Saves the Q_est to a file.
        Args:
            path (str, optional): Path to the file. Defaults to './SARSA_Q_est.npy'.
        """
        np.save(path, self.Q_est)
        
    def load(self, path = './SARSA_Q_est.npy'):
        """ Loads the Q_est from a file.
        Args:
            path (str, optional): Path to the file. Defaults to './SARSA_Q_est.npy'.
        """
        self.Q_est = np.load(path)

class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        """ Initialize the agent.
        Args:
            env (gym.Env): environment to learn from.
            alpha (float): learning rate.
            gamma (float): discount factor.
            epsilon (float): epsilon-greedy policy parameter.
        """
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Sets the Q_est to a matrix of zeros with the shape of (nS, nA)
        self.Q_est = np.array([np.zeros(env.action_space.n) for _ in range(env.observation_space.n)])
        
        
    def get_action(self, state):
        """
        Get an action from Q_est using epsilon-greedy policy. Action is
        optimal with the probability of (1 - epsilon) + epsilon / nA.
        Otherwise, action is selected uniformly at random.
        Args:
            state (int): current state
        Returns:
            action (int): action to take.
        """
        # Selects the action with the highest Q-value with probability 1-epsilon
        if np.random.uniform(0, 1) < 1 - self.epsilon:
            action = np.argmax(self.Q_est[state])
        # Selects a random action from the action space with probability epsilon/nA
        else:
            action = self.env.action_space.sample()
        
        return action
        
    def update(self, state, action, reward, next_state):
        """ Updates the Q_est using the QLearning update rule.
        Q(s, a) <- Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
        Args:
            state (int): state at time t
            action (int): action at time t
            reward (int, float): received reward
            next_state (int): state at time t+1
        """
        # Calculates the temporal difference error (delta)
        delta = reward + self.gamma * np.max(self.Q_est[next_state]) - self.Q_est[state][action]
        # Updates the Q-value for the current state-action pair
        self.Q_est[state][action] += self.alpha * delta
    
    def save(self, path = './QLearning_Q_est.npy'):
        """Saves the Q_est to a file.
        Args:
            path (str, optional): Path to the file. Defaults to './QLearning_Q_est.npy'.
        """
        np.save(path, self.Q_est)
        
    def load(self, path = './QLearning_Q_est.npy'):
        """ Loads the Q_est from a file.
        Args:
            path (str, optional): Path to the file. Defaults to './QLearning_Q_est.npy'.
        """
        print ('PATH:', path)
        self.Q_est = np.load(path)