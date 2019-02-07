'''
Reinforcement Learning by Sutton and Barto
2. Evaluative Feedback
2.3 Softmax Action Selection
'''
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


class Agent(object):
    def __init__(self, n, tau):
        self.n = n
        self.tau = tau
        self.Q = np.zeros(n)
        self.k = np.zeros(n)
    
    def select_action(self):
        weights = np.exp(self.Q/self.tau)
        rand_gen = WeightedRandomGenerator(weights)
        return rand_gen.next()
    
    def accept_reward(self, a, r):
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
    plays_num = 1000
    tasks_num = 2000
    tau = [0.01, 0.1, 0.2, 1] 
     
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    
    for i in range(len(tau)):
        avg_rewards = np.zeros(plays_num)
        optimals = np.zeros(plays_num)
        for j in range(tasks_num):
            env = Environment(n)
            agent = Agent(n, tau[i])
            a_optimal = env.get_optimal_action()
            for p in range(plays_num):
                a = agent.select_action()
                r = env.give_reward(a)
                agent.accept_reward(a, r)
                avg_rewards[p] += r
                if a == a_optimal:
                    optimals[p] += 1
        plays = range(1, plays_num + 1)
        optimals = [100*x/tasks_num for x in optimals]
        avg_rewards = [x/tasks_num for x in avg_rewards]      
        label = 'tau = ' + str(tau[i])
        ax1.plot(plays, avg_rewards, label=label)
        ax2.plot(plays, optimals, label=label)
    
    ax1.set_xlabel('Plays')
    ax1.set_ylabel('Average Reward')
    ax1.legend()
    
    ax2.set_xlabel('Plays')
    ax2.set_ylabel('% Optimal Action')
    ax2.legend()
    
    plt.show()


if __name__ == '__main__':
    main()
