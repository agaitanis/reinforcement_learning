'''
Reinforcement Learning by Sutton and Barto
4. Dynamic Programming
4.3 Policy Iteration
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


def pretty_print_policy(policy):
    row = []
    for s in range(len(policy)):
        actions = []
        if policy[s][UP] > 0:
            actions.append('up')
        if policy[s][DOWN] > 0:
            actions.append('down')
        if policy[s][RIGHT] > 0:
            actions.append('right')
        if policy[s][LEFT] > 0:
            actions.append('left')
        row.append(actions)
        if (s+1)%4 == 0:
            print(row)
            row.clear()


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
    
    def improve_policy(self):
        policy_stable = False

        while not policy_stable:
            policy_stable = True

            self.evaluate_policy()
            
            for s in range(1, self.states_num - 1):
                Q = []
                b = self.policy[s]
                for a in range(self.actions_num):
                    q = 0
                    for next_s in range(self.states_num):
                        q += self.P[s, a, next_s]*(self.R[s, a, next_s] + self.gamma*self.V[next_s])
                    Q.append(q)
                q_max = np.amax(Q)
                best_a = np.argwhere(np.isclose(Q, q_max))
                for a in range(self.actions_num):
                    if np.isclose(Q[a], q_max):
                        self.policy[s, a] = 1/len(best_a)
                    else:
                        self.policy[s, a] = 0
                if (b != self.policy[s]).any():
                    policy_stable = False
            pretty_print_policy(self.policy)



def main():
    states_num = 16
    actions_num = 4
    env = Environment(states_num, actions_num)
    agent = Agent(states_num, actions_num, gamma=1.0, theta=1e-6, P=env.P, R=env.R)
    
    agent.improve_policy()


if __name__ == '__main__':
    main()