import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


MAX_CARS1 = 20
MAX_CARS2 = 20
LAMBDA_RENTALS1 = 3
LAMBDA_RENTALS2 = 4
LAMBDA_RETURNS1 = 3
LAMBDA_RETURNS2 = 2
POISSON_POINTS_NUM = 20
RENTAL_REWARD = 10
MOVE_COST = 2
THETA = 1e-6
GAMMA = 0.9


def poisson(n, l):
    return math.pow(l, n)*math.exp(-l)/math.factorial(n)


def a_to_index(a):
    return a + 5


def index_to_a(a_i):
    return a_i - 5


def calc_probs_and_rewards(cur_s, max_cars, lambda_rentals, lambda_returns):
    states_num = max_cars + 1
    p = np.zeros(states_num)
    r = np.zeros(states_num)
    
    for rentals in range(POISSON_POINTS_NUM):
        p1 = poisson(rentals, lambda_rentals)
        rentals = min(rentals, cur_s)
        reward = RENTAL_REWARD*rentals         
        for returns in range(POISSON_POINTS_NUM):
            p2 = poisson(returns, lambda_returns)
            next_s = min(cur_s - rentals + returns, max_cars)
            p[next_s] += p1*p2
            r[next_s] += p1*p2*reward
            
    for next_s in range(states_num):
        if p[next_s] > 0:
            r[next_s] /= p[next_s]
            
    return p, r


def plot_V(V):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    states1 = range(MAX_CARS1 + 1)
    states2 = range(MAX_CARS2 + 1)
    states1, states2 = np.meshgrid(states1, states2)
    ax.plot_surface(states1, states2, V, cmap=cm.coolwarm)
    ax.set_xlim(0, MAX_CARS1)
    ax.set_ylim(0, MAX_CARS2)
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
    

def main():
    states_num1 = MAX_CARS1 + 1
    states_num2 = MAX_CARS2 + 1
    actions = range(-5, 6)
    V = np.zeros((states_num1, states_num2))
    policy = np.zeros((states_num1, states_num2), dtype=int)
    P = np.zeros((states_num1, states_num2, len(actions), states_num1, states_num2))
    R = np.zeros((states_num1, states_num2, len(actions), states_num1, states_num2))
    
    for s1 in range(states_num1):
        for s2 in range(states_num2):
            for a in actions:
                if a > 0 and a > s1: continue
                if a < 0 and -a > s2: continue
                a_i = a_to_index(a)
                cur_s1 = min(s1 - a, MAX_CARS1)
                cur_s2 = min(s2 + a, MAX_CARS2)
                p1, r1 = calc_probs_and_rewards(cur_s1, MAX_CARS1, LAMBDA_RENTALS1, LAMBDA_RETURNS1)
                p2, r2 = calc_probs_and_rewards(cur_s2, MAX_CARS2, LAMBDA_RENTALS2, LAMBDA_RETURNS2)
                for next_s1 in range(states_num1):
                    for next_s2 in range(states_num2):
                        P[s1, s2, a_i, next_s1, next_s2] = p1[next_s1]*p2[next_s2]
                        R[s1, s2, a_i, next_s1, next_s2] = r1[next_s1] + r2[next_s2] - MOVE_COST*abs(a)
    
    policy_stable = False
    
    while not policy_stable:
        delta = THETA + 1
        k = 0
        while delta > THETA:
            k += 1
            delta = 0
            for s1 in range(states_num1):
                for s2 in range(states_num2):
                    old_v = V[s1, s2]
                    new_v = 0
                    a = policy[s1, s2]
                    a_i = a_to_index(a)
                    for next_s1 in range(states_num1):
                        for next_s2 in range(states_num2):
                            new_v += P[s1, s2, a_i, next_s1, next_s2]*\
                            (R[s1, s2, a_i, next_s1, next_s2] + GAMMA*V[next_s1, next_s2])
                    V[s1, s2] = new_v
                    delta = max(delta, abs(new_v - old_v))
        
        policy_stable = True
        
        for s1 in range(states_num1):
            for s2 in range(states_num2):
                q = []
                b = policy[s1, s2]
                for a in actions:
                    a_i = a_to_index(a)
                    temp_sum = 0
                    for next_s1 in range(states_num1):
                        for next_s2 in range(states_num2):
                            temp_sum += P[s1, s2, a_i, next_s1, next_s2]*\
                            (R[s1, s2, a_i, next_s1, next_s2] + GAMMA*V[next_s1, next_s2])
                    q.append(temp_sum)
                best_a_i = np.argmax(q)
                policy[s1, s2] = index_to_a(best_a_i)
                if b != policy[s1, s2]:
                    policy_stable = False

    plot_V(V)
    plot_policy(policy)

    

if __name__ == '__main__':
    main()
