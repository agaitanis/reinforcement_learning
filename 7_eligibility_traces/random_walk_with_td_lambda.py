# -*- coding: utf-8 -*-
"""
Reinforcement Learning by Sutton and Barto
7. Eligibility Traces
Example 7.3: Random Walk with TD(λ)
"""
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt


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
    def __init__(self, alpha, gamma, l):
        self.alpha = alpha
        self.gamma = gamma
        self.l = l
        self.V = defaultdict(float)
        self.e = defaultdict(float)
        self.states = range(21)


    def start_episode(self):
        self.e.clear()


    def select_action(self):
        return np.random.choice((-1, 1))

    
    def receive_reward(self, s, r, new_s):
        delta = r + self.gamma*self.V[new_s] - self.V[s]
        self.e[s] += 1
        for s in self.states:
            self.V[s] += self.alpha*delta*self.e[s]
            self.e[s] *= self.gamma*self.l
 

def random_walk(l, alpha):
    env = Environment()
    agent = Agent(alpha=alpha, gamma=1.0, l=l)
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
        
        mse = np.mean([(agent.V[s] - true_V[s])**2 for s in env.states[1:-1]])
        error += np.sqrt(mse)
        
    error /= episodes_num
    
    return error


def main():
    np.random.seed(0)
    lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]
    alphas = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25,
              0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    runs_num = 100
    
    plt.figure()
    
    for l in lambdas:
        print('l =', l)
        errors = []
        for alpha in alphas:
            print('alpha =', alpha)
            mean_error = 0
            for run in range(runs_num):
                error = random_walk(l, alpha)
                mean_error += error
            mean_error /= runs_num
            errors.append(mean_error)
        plt.plot(alphas, errors, label="λ = " + str(l))
        print()
    
    plt.xlabel('α')
    plt.ylabel('RMS error')
    plt.ylim(0.25, 0.55)
    plt.legend()


if __name__ == "__main__":
    main()