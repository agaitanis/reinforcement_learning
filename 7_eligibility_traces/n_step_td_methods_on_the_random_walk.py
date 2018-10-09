# -*- coding: utf-8 -*-
"""
Reinforcement Learning by Sutton and Barto
7. Eligibility Traces
7. n-Step TD Prediction
Example 7.1: n-Step TD Methods on the Random Walk
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def update_V_from_buffer(V, buffer, alpha, gamma):
    n = len(buffer)
    R = 0
    for i, (s, r, new_s) in enumerate(buffer):
        R += np.power(gamma, i)*r
    new_s = buffer[-1][2]
    R += np.power(gamma, n)*V[new_s]
    s = buffer[0][0]
    V[s] += alpha*(R - V[s])


def random_walk(n, alpha):
    states = range(21)
    start_state = 10
    end_states = (states[0], states[-1])
    episodes_num = 10
    gamma = 1
    V = np.zeros(len(states))
    rewards = np.zeros(len(states))
    true_V = 0.1*np.arange(21) - 1
    error = 0
    
    rewards[0] = -1
    rewards[-1] = 1
    
    for i in range(episodes_num):
        s = start_state
        buffer = deque(maxlen=n)
        while s not in end_states:
            a = np.random.choice((-1, 1))
            new_s = s + a
            r = rewards[new_s]
            buffer.append((s, r, new_s))
            s = new_s
            if len(buffer) < n:
                continue
            update_V_from_buffer(V, buffer, alpha, gamma)
        while len(buffer) > 1:
            buffer.popleft()
            update_V_from_buffer(V, buffer, alpha, gamma)
        error += np.sqrt(np.mean([np.power(V[s] - true_V[s], 2) for s in states[1:-1]]))
        
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