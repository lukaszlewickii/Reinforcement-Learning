import argparse


import gym
import gym_walking
import numpy as np

parser = argparse.ArgumentParser(description='Runs TD(0) on Walking5-v0.')

parser.add_argument('-e', '--episodes',
                    type=int,
                    default=10,
                    help='Number of episodes. Default: 10'
                   )

parser.add_argument('-g', '--gamma',
                    type=float,
                    default=1.0,
                    help='Gamma parameter for td target formula. Default: 1.0'
                   )

parser.add_argument('-a', '--alpha',
                    type=float,
                    default=0.5,
                    help='Alpha parameter for td target formula. Default: 0.05'
                   )

parser.add_argument('-r', '--render',
                    type=lambda s: s.lower() in ['true', 't', 'yes', '1'],
                    default=True,
                    help='Whether to render or not. Default: True'
                   )


env = gym.make('Walking5-v0')
pi = lambda x: np.random.randint(2) # uniform random policy

def td(pi, env, gamma=1.0, alpha=0.5, n_episodes=10, render=True):
    # ENTER YOUR CODE HERE
    # You should return the vector of state values V
    V = np.array([0] + [0.5] * (env.observation_space.n - 2) + [0])
    
    n_visited = np.array([1] * env.observation_space.n)
    
    for t in range(n_episodes):
        state, info = env.reset()
        n_visited[state] += 1
        terminal = False
        
        visited_states = []
        rewards = []
        
        if render:
            env.render(V)
            
        while not terminal:
            visited_states.append(state)
            action = pi(state)
            next_state, reward, terminal, truncated, info = env.step(action)
            V[state] += alpha * (reward + gamma * V[state + 1] - V[state])
            
            state = next_state
            n_visited[state] += 1
            
            if render:
                env.render(V)
    
    return V

#python td0.py -e 100

if __name__ == "__main__":
    args = parser.parse_args()
    V = td(
        pi, env, 
        gamma=args.gamma, 
        alpha=args.alpha, 
        n_episodes=args.episodes,
        render=args.render
        )
    print ('Final state values: {}'.format(V))