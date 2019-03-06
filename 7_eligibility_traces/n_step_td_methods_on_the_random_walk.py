# -*- coding: utf-8 -*-
"""
Reinforcement Learning by Sutton and Barto
7. Eligibility Traces
7. n-Step TD Prediction
Example 7.1: n-Step TD Methods on the Random Walk
"""
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import deque


class Environment(object):
    def __init__(self):
        self.states = range(21)
        self.start_state = 10
        self.end_states = (self.states[0], self.states[-1])
        self.rewards = np.zeros(len(self.states))
        self.rewards[0] = -1
        self.rewards[-1] = 1
    
    
    def respond_to_action(self, s, a):
        return self.rewards[s + a], s + a


class Agent(object):
    def __init__(self, alpha, gamma, n):
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.V = defaultdict(float)
        self.buffer = deque(maxlen=n)

    
    def select_action(self):
        return np.random.choice((-1, 1))


    def update_V_from_buffer(self):
        n = len(self.buffer)
        R = 0
        for i, (s, r, new_s) in enumerate(self.buffer):
            R += np.power(self.gamma, i)*r
        new_s = self.buffer[-1][2]
        R += np.power(self.gamma, n)*self.V[new_s]
        s = self.buffer[0][0]
        self.V[s] += self.alpha*(R - self.V[s])    

    
    def receive_reward(self, s, r, new_s):
        self.buffer.append((s, r, new_s))
        if len(self.buffer) >= self.n:
            self.update_V_from_buffer()


    def start_episode(self):
        self.buffer.clear()
    
    
    def end_episode(self):
        while len(self.buffer) > 1:
            self.buffer.popleft()
            self.update_V_from_buffer()


def random_walk(n, alpha):
    env = Environment()
    agent = Agent(alpha=alpha, gamma=1.0, n=n)
    episodes_num = 10    
    true_V = 0.1*np.arange(21) - 1
    error = 0

    for i in range(episodes_num):
        s = env.start_state
        agent.start_episode()
        while s not in env.end_states:
            a = agent.select_action()
            r, new_s = env.respond_to_action(s, a)            
            agent.receive_reward(s, r, new_s)
            s = new_s
        agent.end_episode()
        
        mse = np.mean([np.power(agent.V[s] - true_V[s], 2) for s in env.states[1:-1]])
        error += np.sqrt(mse)
        
    error /= episodes_num
    
    return error


def main():
    np.random.seed(0)
    n_steps = [1, 2, 3, 5, 8, 15, 30, 60, 100, 200, 1000]
    alphas = [0, 0.01, 0.02, 0.05,
              0.1, 0.2, 0.3, 0.4, 0.5,
              0.6, 0.7, 0.8, 0.9, 1.0]
    runs_num = 100
    
    plt.figure()
    
    for n in n_steps:
        print('n =', n)
        errors = []
        for alpha in alphas:
            mean_error = 0
            for run in range(runs_num):
                error = random_walk(n, alpha)
                mean_error += error
            mean_error /= runs_num
            errors.append(mean_error)
        plt.plot(alphas, errors, label=str(n))
    
    plt.xlabel('α')
    plt.ylabel('RMS error')
    plt.ylim(0.25, 0.55)
    plt.legend()


if __name__ == "__main__":
    main()