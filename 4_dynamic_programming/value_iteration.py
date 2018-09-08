'''
Reinforcement Learning by Sutton and Barto
4. Dynamic Programming
4.4 Value Iteration
Example 4.3: Gambler's Problem
'''
import numpy as np
import matplotlib.pyplot as plt


def main():
    theta = 1e-9
    gamma = 1
    p = 0.4
    V = np.zeros(101)
    R = np.zeros(101)
    R[100] = 1
    policy = np.zeros(101)
    
    delta = theta + 1

    fig1 = plt.figure()
    ax1 = fig1.gca()
    
    while delta > theta:
        delta = 0
        for s in range(1, 100):
            v = V[s]
            q = []
            for a in range(1, min(s, 100 - s) + 1):
                next_s_to_P = {s + a: p, s - a: 1 - p}
                temp_sum = 0
                for next_s, P in next_s_to_P.items():
                    temp_sum += P*(R[next_s] + gamma*V[next_s])
                q.append(temp_sum)
            max_q = max(q)
            V[s] = max_q
            policy[s] = np.argwhere(abs(q - max_q) < 1e-6)[0] + 1
            delta = max(delta, abs(v - V[s]))    
        ax1.plot(range(1, 100), V[1:100])
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.gca()
    ax2.plot(range(1, 100), policy[1:100])
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake)')
    plt.show()
    

if __name__ == '__main__':
    main()