'''
Reinforcement Learning by Sutton and Barto
2. Evaluative Feedback
2.1 An n-Armed Bandit Problem
'''
import numpy as np
import matplotlib.pyplot as plt


class Agent(object):
    def __init__(self, n, eps):
        self.n = n
        self.eps = eps
        self.Q = np.zeros(n)
        self.k = np.zeros(n)
    
    def select_action(self):
        if np.random.uniform() < self.eps:
            return np.random.randint(0, self.n)
        else:
            return np.argmax(self.Q)
    
    def receive_reward(self, a, r):
        self.k[a] += 1
        self.Q[a] += (r - self.Q[a])/self.k[a]


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
    tasks_num = 2000
    plays_num = 1000
    eps = [0, 0.01, 0.1]
    
    f, (ax1, ax2) = plt.subplots(2, sharex=True)

    for i in range(len(eps)):
        avg_rewards = np.zeros(plays_num)
        optimal_actions = np.zeros(plays_num)
        for j in range(tasks_num):
            env = Environment(n)
            agent = Agent(n, eps[i])
            optimal_action = env.get_optimal_action()
            for p in range(plays_num):
                a = agent.select_action()
                r = env.give_reward(a)
                agent.receive_reward(a, r)
                avg_rewards[p] += r
                if a == optimal_action:
                    optimal_actions[p] += 1
        plays = range(1, plays_num + 1)
        optimal_actions = [100*x/tasks_num for x in optimal_actions]
        avg_rewards = [x/tasks_num for x in avg_rewards]
        label = 'eps = ' + str(eps[i])
        ax1.plot(plays, avg_rewards, label=label)
        ax2.plot(plays, optimal_actions, label=label)
    
    ax1.set_xlabel('Plays')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    
    ax2.set_xlabel('Plays')
    ax2.set_ylabel('% Optimal Action')
    ax2.legend()
    
    plt.show()


if __name__ == '__main__':
    main()
