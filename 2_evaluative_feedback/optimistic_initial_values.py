'''
Reinforcement Learning by Sutton and Barto
2. Evaluative Feedback
2.7 Optimistic Initial Values
'''
import numpy as np
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, n, eps, Q0, alpha):
        self.n = n
        self.eps = eps
        self.alpha = alpha
        self.Q = np.full(n, Q0)
        self.k = np.zeros(n)
    
    def select_action(self):
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.n)
        else:
            return np.argmax(self.Q)
    
    def accept_reward(self, a, r):
        self.Q[a] += self.alpha*(r - self.Q[a])


class Environment(object):
    def __init__(self, n):
        self.Q_star = np.random.normal(size=n)
        self.optimal_action = np.argmax(self.Q_star)
    
    def give_reward(self, a):
        return np.random.normal(self.Q_star[a], 1)
    
    def get_optimal_action(self):
        return self.optimal_action


def main():
    np.random.seed(0)
    n = 10
    plays_num = 1000
    tasks_num = 2000
    eps = [0., 0.1]
    Q0 = [5., 0.]
    alpha = 0.1
    labels = ['Optimistic greedy\nQ0 = 5, ε = 0',
              'Realistic ε-greedy\nQ0 = 0, ε = 0.1']    
    plays = range(1, plays_num + 1)

    for i in range(len(eps)):
        optimal_actions = plays_num*[0]
        for j in range(tasks_num):
            env = Environment(n)
            agent = Agent(n, eps[i], Q0[i], alpha)
            optimal_action = env.get_optimal_action()
            for p in range(plays_num):
                a = agent.select_action()
                r = env.give_reward(a)
                agent.accept_reward(a, r)
                if a == optimal_action:
                    optimal_actions[p] += 1
        optimal_actions = [100*x/tasks_num for x in optimal_actions]
        plt.plot(plays, optimal_actions, label=labels[i])

    plt.xlabel('Plays')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
