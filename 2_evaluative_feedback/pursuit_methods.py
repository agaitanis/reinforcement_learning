'''
Reinforcement Learning by Sutton and Barto
2. Evaluative Feedback
2.9 Pursuit Methods
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


class Agent1(object):
    def __init__(self, n, beta):
        self.n = n
        self.beta = beta
        self.Q = np.zeros(n)
        self.k = np.zeros(n)
        self.probs = np.full(n, 1/n)
    
    def select_action(self):
        rand_gen = WeightedRandomGenerator(self.probs)
        return rand_gen.next()
    
    def receive_reward(self, a, r):
        self.k[a] += 1
        self.Q[a] += (r - self.Q[a])/self.k[a]
        a_star = np.argmax(self.Q)
        self.probs[a_star] += self.beta*(1 - self.probs[a_star])
        for a in range(self.n):
            if a != a_star:
                self.probs[a] += self.beta*(0 - self.probs[a])


class Agent2(object):
    def __init__(self, n, beta, alpha):
        self.beta = beta
        self.alpha = alpha
        self.p = np.zeros(n)
        self.r_mean = 0
    
    def select_action(self):
        weights = np.exp(self.p)
        rand_gen = WeightedRandomGenerator(weights)
        return rand_gen.next()
    
    def receive_reward(self, a, r):
        self.p[a] += self.beta*(r - self.r_mean)
        self.r_mean += self.alpha*(r - self.r_mean)


class Agent3(object):
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
    plays_num = 1000
    tasks_num = 2000
    labels = ['pursuit',
              'reinforcement comparison',
              'ε-greedy\nε = 0.1, α = 1/k']
    
    plays = range(1, plays_num + 1)
    
    for i in range(3):
        optimal_actions = np.zeros(plays_num)
        for j in range(tasks_num):
            env = Environment(n)
            if i == 0:
                agent = Agent1(n, beta=0.01)
            elif i == 1:
                agent = Agent2(n, beta=0.1, alpha=0.1)
            else:
                agent = Agent3(n, eps=0.1)
            optimal_action = env.get_optimal_action()
            for p in range(plays_num):
                a = agent.select_action()
                r = env.give_reward(a)
                agent.receive_reward(a, r)
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
