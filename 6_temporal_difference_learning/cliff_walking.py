# -*- coding: utf-8 -*-
"""
Reinforcement Learning by Sutton and Barto
6. Temporal-Difference Learning
6.2 Q-Learning: Off-Policy TD Control
Example 6.6: Cliff Walking
"""
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import scipy


class CliffWalking:
    def __init__(self):
        self.eps = 0.1
        self.alpha = 0.5
        self.gamma = 1
        self.episodes_num = 500
        self.grid_size = (4, 12)
        self.start_state = (3, 0)
        self.end_state = (3, 11)
        self.cliff = [(3, x) for x in range(1, 11)]
        self.moves = ((-1, 0), (1, 0), (0, 1), (0, -1))
        self.action_str = ('u', 'd', 'r', 'l')
        self.actions = range(len(self.moves))


    def print_greedy_policy(self, Q):
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                s = (i, j)
                a = np.argmax(Q[s])
                print(self.action_str[a], end=' ')
            print()
        print()

    
    def get_next_action(self, s, Q):
        if np.random.uniform() < self.eps:
            return np.random.choice(self.actions)
        else:
            return np.argmax(Q[s])


    def get_next_state(self, s, a):
        new_s = tuple(np.clip(np.array(s) + np.array(self.moves[a]), (0, 0), self.grid_size))
        if new_s in self.cliff:
            return self.start_state, -100
        else:
            return tuple(new_s), -1

    
    def q_learning(self):
        np.random.seed(0)
        Q = defaultdict(lambda: len(self.actions)*[0])

        for i in range(self.episodes_num):
            s = self.start_state
            while s != self.end_state:
                a = self.get_next_action(s, Q)
                new_s, r = self.get_next_state(s, a)
                Q[s][a] += self.alpha*(r + self.gamma*max(Q[new_s]) - Q[s][a])
                s = new_s
        print("Q-Learning")
        self.print_greedy_policy(Q)


    def sarsa(self):
        np.random.seed(0)
        Q = defaultdict(lambda: len(self.actions)*[0])

        for i in range(self.episodes_num):
            s = self.start_state
            a = self.get_next_action(s, Q)
            while s != self.end_state:
                new_s, r = self.get_next_state(s, a)
                new_a = self.get_next_action(new_s, Q)
                Q[s][a] += self.alpha*(r + self.gamma*Q[new_s][new_a] - Q[s][a])
                s = new_s
                a = new_a
        print("Sarsa")
        self.print_greedy_policy(Q)


def main():
    cw = CliffWalking()   
    cw.q_learning()
    cw.sarsa()


if __name__ == "__main__":
    main()