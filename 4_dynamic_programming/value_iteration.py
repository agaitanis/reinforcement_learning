'''
Reinforcement Learning by Sutton and Barto
4. Dynamic Programming
4.4 Value Iteration
Example 4.3: Gambler's Problem
'''
import numpy as np


def main():
    theta = 1e-18
    p = 0.4
    states_num = 101
    gamma = 1
    V = np.zeros(states_num)
    R = np.zeros(states_num)
    R[100] = 1
    
    delta = theta + 1
    k = 0
    
    while delta > theta:
        print("k =", k)
        print(V)
        k += 1
        delta = 0
        for s in range(1, states_num - 1):
            v = V[s]
            q = []
            for a in range(1, min(s, 100 - s) + 1):
                win_s = s + a
                lose_s = s - a
                next_states = (win_s, lose_s)
                P = {win_s : p, lose_s : 1 - p}
                q_a = 0
                for next_s in next_states:
                    q_a += P[next_s]*(R[next_s] + gamma*V[next_s])
                q.append(q_a)
            V[s] = max(q)
            delta = max(delta, abs(v - V[s]))
        

if __name__ == '__main__':
    main()