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


def random_boltzmann(p):
    weights = [np.exp(p[a]) for a in range(len(p))]
    rand_gen = WeightedRandomGenerator(weights)
    return rand_gen.next()


def main():
    np.random.seed(0)
    n = 10
    plays_num = 1000
    tasks_num = 2000
    alpha = 0.1
    beta = 0.1
    eps = 0.1
    labels = ['reinforcement comparison',
              'ε-greedy\nε = 0.1, α = 1/k',
              'ε-greedy\nε = 0.1, α = 0.1']
    
    plays = range(1, plays_num + 1)
    
    for i in range(3):
        optimals = plays_num*[0]
        for j in range(tasks_num):
            q_star = np.random.normal(size=n)
            a_optimal = np.argmax(q_star)
            if i == 0: # reinforcement comparison
                p = n*[0]
                r_mean = 0
                for t in range(plays_num):
                    a = random_boltzmann(p)
                    r = np.random.normal(q_star[a], 1)
                    p[a] += beta*(r - r_mean)
                    r_mean += alpha*(r - r_mean)
                    if a == a_optimal: optimals[t] += 1
            else: # ε-greedy
                q = n*[0]
                k = n*[0]
                for t in range(plays_num):
                    if np.random.uniform() < eps:
                        a = np.random.randint(0, n)
                    else:
                        a = np.argmax(q)
                    r = np.random.normal(q_star[a], 1)
                    if i == 1:
                        k[a] += 1
                        q[a] += (r - q[a])/k[a]
                    else:
                        q[a] += alpha*(r - q[a])
                    if a == a_optimal: optimals[t] += 1          

        optimals = [100*x/tasks_num for x in optimals]
        plt.plot(plays, optimals, label=labels[i])
    
    plt.xlabel('Plays')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
