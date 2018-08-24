'''
Reinforcement Learning by Sutton and Barto
2. Evaluative Feedback
2.7 Optimistic Initial Values
'''
import numpy as np
import matplotlib.pyplot as plt


def main():
    np.random.seed(0)
    n = 10
    plays_num = 1000
    tasks_num = 2000
    eps = [0, 0.1]
    q0 = [5, 0]
    alpha = 0.1
    labels = ['Optimistic greedy\nQ0 = 5, ε = 0',
              'Realistic ε-greedy\nQ0 = 0, ε = 0.1']
    
    plays = range(1, plays_num + 1)

    for i in range(len(eps)):
        optimals = plays_num*[0]
        for j in range(tasks_num):
            q_star = np.random.normal(size=n)
            a_optimal = np.argmax(q_star)
            q = n*[q0[i]]
            for p in range(plays_num):
                if np.random.uniform() < eps[i]:
                    a = np.random.randint(0, n)
                else:
                    a = np.argmax(q)
                r = np.random.normal(q_star[a], 1)
                q[a] += alpha*(r - q[a])
                if a == a_optimal:
                    optimals[p] += 1
        optimals = [100*x/tasks_num for x in optimals]
        plt.plot(plays, optimals, label=labels[i])

    plt.xlabel('Plays')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
