'''
Reinforcement Learning by Sutton and Barto
4. Dynamic Programming
4.1 Policy Evaluation
Example 4.1: 4x4 Gridworld
'''
import numpy as np

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3

            
def get_next_state(s, a):
    if a == UP:
        if s >= 4: return s - 4
    elif a == DOWN:
        if s < 12: return s + 4
    elif a == RIGHT:
        if (s + 1)%4 != 0: return s + 1
    elif a == LEFT:
        if s%4 != 0: return s - 1
    return s


class Environment(object):
    def __init__(self, states_num, actions_num):
        self.P = np.zeros((states_num, actions_num, states_num))
        self.R = np.full((states_num, actions_num, states_num), -1)
        
        for s in range(1, states_num - 1):
            for a in range(actions_num):
                next_s = get_next_state(s, a)
                self.P[s, a, next_s] = 1


class Agent(object):
    def __init__(self, states_num, actions_num, gamma, theta, P, R):
        self.states_num = states_num
        self.actions_num = actions_num
        self.gamma = gamma
        self.theta = theta
        self.P = P
        self.R = R
        self.V = np.zeros(states_num)
        self.policy = np.zeros((states_num, actions_num))
        for s in range(1, states_num - 1):
            for a in range(actions_num):
                self.policy[s, a] = 0.25
    
    def evaluate_policy(self):     
        delta = self.theta + 1
        k = 0
    
        while delta > self.theta:
            print("k =", k)
            print(self.V.reshape(4, 4))
            k += 1
            delta = 0
            for s in range(self.states_num):
                old_v = self.V[s]
                new_v = 0
                for a in range(self.actions_num):
                    q = 0
                    for next_s in range(self.states_num):
                        q += self.P[s, a, next_s]*(self.R[s, a, next_s] + self.gamma*self.V[next_s])
                    new_v += self.policy[s, a]*q
                self.V[s] = new_v
                delta = max(delta, abs(new_v - old_v))



def main():
    states_num = 16
    actions_num = 4
    env = Environment(states_num, actions_num)
    agent = Agent(states_num, actions_num, gamma=1.0, theta=1e-6, P=env.P, R=env.R)

    agent.evaluate_policy()


if __name__ == '__main__':
    main()