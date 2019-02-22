'''
Reinforcement Learning by Sutton and Barto
4. Dynamic Programming
4.3 Policy Iteration
Example 4.2: Jack's Car Rental
'''
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


def poisson(n, l):
    return math.pow(l, n)*math.exp(-l)/math.factorial(n)


def a_to_index(a):
    return a + 5


def index_to_a(a_i):
    return a_i - 5


def calc_probs_and_rewards(cur_s, max_cars, lambda_rentals, lambda_returns,
                           poisson_points_num, rental_reward):
    states_num = max_cars + 1
    p = np.zeros(states_num)
    r = np.zeros(states_num)
    
    for rentals in range(poisson_points_num):
        p1 = poisson(rentals, lambda_rentals)
        rentals = min(rentals, cur_s)
        reward = rental_reward*rentals         
        for returns in range(poisson_points_num):
            p2 = poisson(returns, lambda_returns)
            next_s = min(cur_s - rentals + returns, max_cars)
            p[next_s] += p1*p2
            r[next_s] += p1*p2*reward
            
    for next_s in range(states_num):
        if p[next_s] > 0:
            r[next_s] /= p[next_s]
            
    return p, r


def plot_V(V, max_cars1, max_cars2):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    states1 = range(max_cars1 + 1)
    states2 = range(max_cars2 + 1)
    states1, states2 = np.meshgrid(states1, states2)
    ax.plot_surface(states1, states2, V, cmap=cm.coolwarm)
    ax.set_xlim(0, max_cars1)
    ax.set_ylim(0, max_cars2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('V')
    plt.show()


def plot_policy(policy):
    fig = plt.figure()
    ax = fig.gca()
    ax.matshow(policy, cmap=cm.coolwarm)
    for (i, j), z in np.ndenumerate(policy):
        ax.text(j, i, str(z), ha='center', va='center')
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    ax.set_title('π')
    plt.show()


class Environment(object):
    def __init__(self, max_cars1, max_cars2, actions,
                 lambda_rentals1, lambda_rentals2,
                 lambda_returns1, lambda_returns2,
                 poisson_points_num, rental_reward,
                 move_cost):
        states_num1 = max_cars1 + 1
        states_num2 = max_cars2 + 1
        self.P = np.zeros((states_num1, states_num2, len(actions), states_num1, states_num2))
        self.R = np.zeros((states_num1, states_num2, len(actions), states_num1, states_num2))
    
        for s1 in range(states_num1):
            for s2 in range(states_num2):
                for a in actions:
                    if a > 0 and a > s1: continue
                    if a < 0 and -a > s2: continue
                    a_i = a_to_index(a)
                    cur_s1 = min(s1 - a, max_cars1)
                    cur_s2 = min(s2 + a, max_cars2)
                    p1, r1 = calc_probs_and_rewards(cur_s1, max_cars1,
                                                    lambda_rentals1, lambda_returns1,
                                                    poisson_points_num, rental_reward)
                    p2, r2 = calc_probs_and_rewards(cur_s2, max_cars2,
                                                    lambda_rentals2, lambda_returns2,
                                                    poisson_points_num, rental_reward)
                    for next_s1 in range(states_num1):
                        for next_s2 in range(states_num2):
                            self.P[s1, s2, a_i, next_s1, next_s2] = p1[next_s1]*p2[next_s2]
                            self.R[s1, s2, a_i, next_s1, next_s2] = r1[next_s1] + r2[next_s2] - move_cost*abs(a)


class Agent(object):
    def __init__(self, max_cars1, max_cars2, actions, gamma, theta, P, R):
        self.states_num1 = max_cars1 + 1
        self.states_num2 = max_cars2 + 1
        self.actions = actions
        self.gamma = gamma
        self.theta = theta
        self.P = P
        self.R = R
        self.V = np.zeros((self.states_num1, self.states_num2))
        self.policy = np.zeros((self.states_num1, self.states_num2), dtype=int)

    
    def evaluate_policy(self):
        delta = self.theta + 1
        
        while delta > self.theta:
            delta = 0
            for s1 in range(self.states_num1):
                for s2 in range(self.states_num2):
                    old_v = self.V[s1, s2]
                    new_v = 0
                    a = self.policy[s1, s2]
                    a_i = a_to_index(a)
                    for next_s1 in range(self.states_num1):
                        for next_s2 in range(self.states_num2):
                            new_v += self.P[s1, s2, a_i, next_s1, next_s2]*\
                            (self.R[s1, s2, a_i, next_s1, next_s2] + self.gamma*self.V[next_s1, next_s2])
                    self.V[s1, s2] = new_v
                    delta = max(delta, abs(new_v - old_v))


    def improve_policy(self):
        policy_stable = False
        
        while not policy_stable:
            self.evaluate_policy()
            
            policy_stable = True
            
            for s1 in range(self.states_num1):
                for s2 in range(self.states_num2):
                    q = []
                    b = self.policy[s1, s2]
                    for a in self.actions:
                        a_i = a_to_index(a)
                        temp_sum = 0
                        for next_s1 in range(self.states_num1):
                            for next_s2 in range(self.states_num2):
                                temp_sum += self.P[s1, s2, a_i, next_s1, next_s2]*\
                                (self.R[s1, s2, a_i, next_s1, next_s2] + self.gamma*self.V[next_s1, next_s2])
                        q.append(temp_sum)
                    best_a_i = np.argmax(q)
                    self.policy[s1, s2] = index_to_a(best_a_i)
                    if b != self.policy[s1, s2]:
                        policy_stable = False


def main():
    max_cars1 = 20
    max_cars2 = 20
    actions = range(-5, 6)
    env = Environment(max_cars1, max_cars2, actions,
                      lambda_rentals1=3, lambda_rentals2=4,
                      lambda_returns1=3, lambda_returns2=2,
                      poisson_points_num=20, rental_reward=10,
                      move_cost=2)
    agent = Agent(max_cars1, max_cars2, actions,
                  gamma=0.9, theta=1e-6, P=env.P, R=env.R)
    
    agent.improve_policy()

    plot_V(agent.V, max_cars1, max_cars2)
    plot_policy(agent.policy)

    

if __name__ == '__main__':
    main()
