import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

n = 10
plays_num = 1000
tasks_num = 2000
eps = [0, 0.01, 0.1]
labels = ['eps = ' + str(x) for x in eps]
colors = ['red', 'green', 'blue']

plays = range(1, plays_num + 1)

for i in range(len(eps)):
    r_means = plays_num*[0]
    for j in range(tasks_num):
        q_star = np.random.normal(size=n)
        q = n*[0]
        k = n*[0]
        for p in range(plays_num):
            if np.random.uniform() < eps[i]:
                a = np.random.randint(0, n)
            else:
                a = q.index(max(q))
            r = np.random.normal(q_star[a], 1) 
            q[a] += (r - q[a])/(k[a] + 1)
            k[a] += 1
            r_means[p] += (r - r_means[p])/(j + 1)
    plt.plot(plays, r_means, color=colors[i], label=labels[i])

plt.xlabel('Plays')
plt.ylabel('Average Reward')
plt.legend()
plt.show()
