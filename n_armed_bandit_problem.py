import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(0)
    n = 10
    plays_num = 1000
    tasks_num = 2000
    eps = [0, 0.01, 0.1]
    labels = ['eps = ' + str(x) for x in eps]
    
    plays = range(1, plays_num + 1)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)

    for i in range(len(eps)):
        r_means = np.zeros(plays_num)
        optimals = np.zeros(plays_num)
        for j in range(tasks_num):
            q_star = np.random.normal(size=n)
            a_optimal = np.argmax(q_star)
            q = np.zeros(n)
            k = np.zeros(n)
            for p in range(plays_num):
                if np.random.uniform() < eps[i]:
                    a = np.random.randint(0, n)
                else:
                    a = np.argmax(q)
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


if __name__ == '__main__':
    main()
