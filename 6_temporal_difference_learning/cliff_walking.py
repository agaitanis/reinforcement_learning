# -*- coding: utf-8 -*-
"""
Reinforcement Learning by Sutton and Barto
6. Temporal-Difference Learning
6.2 Q-Learning: Off-Policy TD Control
Example 6.6: Cliff Walking
"""
import numpy as np
from collections import defaultdict


def print_greedy_policy(name, Q, grid_size, action_to_str):
    print(name)
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            s = (i, j)
            a = np.argmax(Q[s])
            print(action_to_str[a], end=' ')
        print()
    print()


class Environment(object):
    def __init__(self):
        self.grid_size = (4, 12)
        self.start_state = (3, 0)
        self.end_state = (3, 11)
        self.cliff = [(3, x) for x in range(1, 11)]


    def respond_to_move(self, s, move):
        new_s = tuple(np.clip(np.array(s) + np.array(move), (0, 0), self.grid_size))
        if new_s in self.cliff:
            return self.start_state, -100
        else:
            return tuple(new_s), -1


class AgentQ(object):
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
    
    
    def improve_policy(self, s, a, r, new_s):
        self.Q[s][a] += self.alpha*(r + self.gamma*max(self.Q[new_s]) - self.Q[s][a])
    
    
def q_learning(episodes_num, alpha, gamma, eps):
    np.random.seed(0)
    env = Environment()
    agent = AgentQ(alpha, gamma, eps)

    for i in range(episodes_num):
        s = env.start_state
        while s != env.end_state:
            a = agent.select_action(s)
            move = agent.action_to_move[a]
            new_s, r = env.respond_to_move(s, move)
            agent.improve_policy(s, a, r, new_s)
            s = new_s

    print_greedy_policy("Q-Learning", agent.Q, env.grid_size, agent.action_to_str)


class AgentSarsa(object):
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


def sarsa(episodes_num, alpha, gamma, eps):
    np.random.seed(0)
    env = Environment()
    agent = AgentSarsa(alpha, gamma, eps)

    for i in range(episodes_num):
        s = env.start_state
        a = agent.select_action(s)
        while s != env.end_state:
            move = agent.action_to_move[a]
            new_s, r = env.respond_to_move(s, move)
            new_a = agent.select_action(new_s)
            agent.improve_policy(s, a, r, new_s, new_a)
            s, a, = new_s, new_a

    print_greedy_policy("Sarsa", agent.Q, env.grid_size, agent.action_to_str)


def main():
    episodes_num = 500
    alpha = 0.5
    gamma = 1.0
    eps = 0.1
    q_learning(episodes_num, alpha, gamma, eps)
    sarsa(episodes_num, alpha, gamma, eps)
    

if __name__ == "__main__":
    main()