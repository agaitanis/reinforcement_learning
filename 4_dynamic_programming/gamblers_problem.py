'''
Reinforcement Learning by Sutton and Barto
4. Dynamic Programming
4.4 Value Iteration
Example 4.3: Gambler's Problem
'''
import numpy as np
import matplotlib.pyplot as plt


class Environment(object):
    def __init__(self, states_num, p):
        self.p = p
        self.R = np.zeros(states_num)
        self.R[100] = 1


class Agent(object):
    def __init__(self, states_num, gamma, theta, p, R):
        self.gamma = gamma
        self.theta = theta
        self.p = p
        self.R = R
        self.V = np.zeros(states_num)
        self.policy = np.zeros(states_num)
    
    def improve_policy(self):
        delta = self.theta + 1
    
        fig1 = plt.figure()
        ax1 = fig1.gca()
        
        while delta > self.theta:
            delta = 0
            for s in range(1, 100):
                v = self.V[s]
                q = []
                for a in range(1, min(s, 100 - s) + 1):
                    next_s_to_P = {s + a: self.p, s - a: 1 - self.p}
                    temp_sum = 0
                    for next_s, P in next_s_to_P.items():
                        temp_sum += P*(self.R[next_s] + self.gamma*self.V[next_s])
                    q.append(temp_sum)
                max_q = max(q)
                self.V[s] = max_q
                self.policy[s] = np.argwhere(abs(q - max_q) < 1e-6)[0] + 1
                delta = max(delta, abs(v - self.V[s]))    
            ax1.plot(range(1, 100), self.V[1:100])
        plt.xlabel('Capital')
        plt.ylabel('Value estimates')
        plt.show()
        
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.plot(range(1, 100), self.policy[1:100])
        plt.xlabel('Capital')
        plt.ylabel('Final policy (stake)')
        plt.show()


def main():
    states_num = 101
    env = Environment(states_num, p=0.4)
    agent = Agent(states_num, gamma=1.0, theta=1e-9, p=env.p, R=env.R)
    
    agent.improve_policy()


if __name__ == '__main__':
    main()