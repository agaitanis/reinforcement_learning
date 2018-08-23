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
POISSON_POINTS_NUM = 15
RENTAL_REWARD = 10
MOVE_COST = 2
THETA = 1e-6
GAMMA = 0.9


def poisson(n, l):
    return math.pow(l, n)*math.exp(-l)/math.factorial(n)


def a_to_index(a):
    return a + 5


def calc_probs_and_rewards(cur_s, max_cars, lambda_rentals, lambda_returns):
    states_num = max_cars + 1
    p = np.zeros(states_num)
    r = np.zeros(states_num)
    for rentals in range(POISSON_POINTS_NUM):
        p1 = poisson(rentals, lambda_rentals)
        possible_rentals = min(rentals, cur_s)
        reward = RENTAL_REWARD*possible_rentals
        s_after_rental = cur_s - possible_rentals
        for returns in range(POISSON_POINTS_NUM):
            p2 = poisson(returns, lambda_returns)
            n_s = min(s_after_rental + returns, max_cars)
            prob = p1*p2
            p[n_s] += prob
            r[n_s] += prob*reward
    return p, r


def plot_V(V):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    states1 = range(MAX_CARS1 + 1)
    states2 = range(MAX_CARS2 + 1)
    states1, states2 = np.meshgrid(states1, states2)
    surf = ax.plot_surface(states1, states2, V, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0, MAX_CARS1)
    ax.set_ylim(0, MAX_CARS2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))   
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
                a_i = a_to_index(a)
                cur_s1 = np.clip(s1 - a, 0, MAX_CARS1)
                cur_s2 = np.clip(s2 + a, 0, MAX_CARS2)
                p1, r1 = calc_probs_and_rewards(cur_s1, MAX_CARS1, LAMBDA_RENTALS1, LAMBDA_RETURNS1)
                p2, r2 = calc_probs_and_rewards(cur_s2, MAX_CARS2, LAMBDA_RENTALS2, LAMBDA_RETURNS2)
                for n_s1 in range(states_num1):
                    for n_s2 in range(states_num2):
                        P[s1, s2, a_i, n_s1, n_s2] = p1[n_s1]*p2[n_s2]
                        R[s1, s2, a_i, n_s1, n_s2] = r1[n_s1] + r2[n_s2] - MOVE_COST*abs(a)

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
    plot_V(V)



if __name__ == '__main__':
    main()
