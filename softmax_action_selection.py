import numpy as np
import matplotlib.pyplot as plt
import bisect

np.random.seed(0)

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


def random_boltzmann(q, T):
    weights = [np.exp(q[a]/T) for a in range(len(q))]
    rand_gen = WeightedRandomGenerator(weights)
    return rand_gen.next()


n = 10
plays_num = 1000
tasks_num = 2000
tau = [0.01, 0.1, 0.2, 1]
labels = ['tau = ' + str(x) for x in tau]

plays = range(1, plays_num + 1)
f, (ax1, ax2) = plt.subplots(2, sharex=True)

for i in range(len(tau)):
    r_means = np.zeros(plays_num)
    optimals = np.zeros(plays_num)
    for j in range(tasks_num):
        q_star = np.random.normal(size=n)
        a_optimal = np.argmax(q_star)
        q = np.zeros(n)
        k = np.zeros(n)
        for p in range(plays_num):
            a = random_boltzmann(q, tau[i])
            r = np.random.normal(q_star[a], 1)
            q[a] += (r - q[a])/(k[a] + 1)
            k[a] += 1
            r_means[p] += r
            if a == a_optimal:
                optimals[p] += 1
    optimals = [100*x/tasks_num for x in optimals]
    r_means = [x/tasks_num for x in r_means]
    ax1.plot(plays, r_means, label=labels[i])
    ax2.plot(plays, optimals, label=labels[i])

ax1.set_xlabel('Plays')
ax1.set_ylabel('Average Reward')
ax1.legend()

ax2.set_xlabel('Plays')
ax2.set_ylabel('% Optimal Action')
ax2.legend()

plt.show()
