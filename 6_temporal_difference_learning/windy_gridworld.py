# -*- coding: utf-8 -*-
'''
Reinforcement Learning by Sutton and Barto
6. Temporal-Difference Learning
6.4 Sarsa: On-Policy TD Control
Example 6.5: Windy Gridworld
'''
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


class Environment(object):
    def __init__(self):
        self.grid_size = (7, 10)
        self.start_state = (3, 0)
        self.end_state = (3, 7)
        self.wind = (0, 0, 0, 1, 1, 1, 2, 2, 1, 0)


    def respond_to_move(self, s, move):
        r = -1
        new_s = (np.clip(s[0] + move[0] - self.wind[s[1]], 0, self.grid_size[0] - 1), 
                 np.clip(s[1] + move[1], 0, self.grid_size[1] - 1))
        return r, new_s
    

class Agent(object):
    def __init__(self, alpha, gamma, eps):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.action_to_move = ((-1, 0), (1, 0), (0, 1), (0, -1))
        self.action_to_str = ('u', 'd', 'r', 'l')
        self.actions = range(len(self.action_to_move))
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))


    def select_action(self, s):
        if np.random.uniform() < self.eps:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.Q[s])


    def improve_policy(self, s, a, r, new_s, new_a):
        self.Q[s][a] += self.alpha*(r + self.gamma*self.Q[new_s][new_a] - self.Q[s][a])

    
    def convert_action_to_move(self, a):
        return self.action_to_move[a]


    def print_greedy_policy(self, grid_size):
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                s = (i, j)
                a = np.argmax(self.Q[s])
                print(self.action_to_str[a], end=' ')
            print()
        print()    


def main():
    np.random.seed(0)
    env = Environment()
    agent = Agent(alpha=0.4, gamma=1.0, eps=0.1)  
    episodes_num = 170
    steps = 0
    episodes = []
    
    for i in range(episodes_num):
        s = env.start_state
        a = agent.select_action(s)
        while s != env.end_state:
            move = agent.convert_action_to_move(a)
            r, new_s = env.respond_to_move(s, move)
            new_a = agent.select_action(new_s)
            agent.improve_policy(s, a, r, new_s, new_a)
            s, a = new_s, new_a
            steps += 1
            episodes.append(i)
            
    agent.print_greedy_policy(env.grid_size)
    plt.plot(range(steps), episodes)
    plt.xlabel("Time steps")
    plt.ylabel("Episodes")
    


if __name__ == "__main__":
    main()