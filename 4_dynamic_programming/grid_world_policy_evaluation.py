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
    def __init__(self, env, states_num, actions_num, gamma, theta):
        self.env = env
        self.states_num = states_num
        self.actions_num = actions_num
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros(states_num)
        self.policy = np.zeros((states_num, actions_num))
        for s in range(1, states_num - 1):
            for a in range(actions_num):
                self.policy[s, a] = 0.25
    
    def evaluate_policy(self):     
        states_num = self.states_num
        actions_num = self.actions_num
        theta = self.theta
        V = self.V
        gamma = self.gamma
        P = self.env.P
        R = self.env.R
        delta = self.theta + 1
        policy = self.policy
        k = 0
    
        while delta > theta:
            print("k =", k)
            print(V.reshape(4, 4))
            k += 1
            delta = 0
            for s in range(states_num):
                old_v = V[s]
                new_v = 0
                for a in range(actions_num):
                    temp_sum = 0
                    for next_s in range(states_num):
                        temp_sum += P[s, a, next_s]*(R[s, a, next_s] + gamma*V[next_s])
                    new_v += policy[s, a]*temp_sum
                V[s] = new_v
                delta = max(delta, abs(new_v - old_v))



def main():
    env = Environment(states_num=16, actions_num=4)
    agent = Agent(env=env, states_num=16, actions_num=4, gamma=1.0, theta=1e-6)

    agent.evaluate_policy()


if __name__ == '__main__':
    main()