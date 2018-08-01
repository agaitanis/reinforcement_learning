import numpy as np
import matplotlib.pyplot as plt
import bisect


class WeightedRandomGenerator(object):
    def __init__(self, weights):
        self.totals = []
        running_total = 0

        for w in weights:
            running_total += w
            self.totals.append(running_total)

    def next(self):
        rnd = np.random.uniform() * self.totals[-1]
        return bisect.bisect_right(self.totals, rnd)


def rand_by_weights(weights):
    rand_gen = WeightedRandomGenerator(weights)
    return rand_gen.next()


def main():
    np.random.seed(0)
    n = 10
    plays_num = 1000
    tasks_num = 2000
    labels = ['pursuit',
              'reinforcement comparison',
              'ε-greedy\nε = 0.1, α = 1/k']
    
    plays = range(1, plays_num + 1)
    
    for i in range(3):
        optimals = plays_num*[0]
        for j in range(tasks_num):
            q_star = np.random.normal(size=n)
            a_optimal = np.argmax(q_star)
            if i == 0: # pursuit
                beta = 0.01
                k = n*[0]
                q = n*[0]
                p = n*[1/n]
                for t in range(plays_num):
                    a = rand_by_weights(p)
                    r = np.random.normal(q_star[a], 1)
                    k[a] += 1
                    q[a] += (r - q[a])/k[a]
                    a_star = np.argmax(q)
                    p[a_star] += beta*(1 - p[a_star])
                    for ai in range(n):
                        if ai == a_star: continue
                        p[ai] += beta*(0 - p[ai])
                    if a == a_optimal: optimals[t] += 1
            elif i == 1: # reinforcement comparison
                alpha = 0.1
                beta = 0.1
                p = n*[0]
                r_mean = 0
                for t in range(plays_num):
                    a = rand_by_weights(np.exp(p))
                    r = np.random.normal(q_star[a], 1)
                    p[a] += beta*(r - r_mean)
                    r_mean += alpha*(r - r_mean)
                    if a == a_optimal: optimals[t] += 1
            else: # ε-greedy
                eps = 0.1
                q = n*[0]
                k = n*[0]
                for t in range(plays_num):
                    if np.random.uniform() < eps:
                        a = np.random.randint(0, n)
                    else:
                        a = np.argmax(q)
                    r = np.random.normal(q_star[a], 1)
                    k[a] += 1
                    q[a] += (r - q[a])/k[a]
                    if a == a_optimal: optimals[t] += 1          

        optimals = [100*x/tasks_num for x in optimals]
        plt.plot(plays, optimals, label=labels[i])
    
    plt.xlabel('Plays')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
