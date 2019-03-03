'''
Reinforcement Learning by Sutton and Barto
6. Temporal-Difference Learning
6.2 Advantages of TD Prediction Methods
Example 6.2: Random Walk
'''
import numpy as np
import matplotlib.pyplot as plt


class AgentTD(object):
    def __init__(self, alpha, gamma):
        self.alpha = alpha
        self.gamma = gamma
        self.V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 0]
        
    def evaluate_policy(self, s, r, next_s):
        self.V[s] += self.alpha*(r + self.gamma*self.V[next_s] - self.V[s])


class AgentMC(object):
    def __init__(self, alpha):
        self.alpha = alpha
        self.V = [0, 0.5, 0.5, 0.5, 0.5, 0.5, 1]
        
    def evaluate_policy(self, episode_states, R):
            for s in episode_states:
                self.V[s] += self.alpha*(R - self.V[s])


def td_plot_V():
    # TD(0)
    true_V = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
    initial_state = 3
    end_states = (0, 6)
    win_state = 6
    alpha = 0.1
    gamma = 1
    episodes_num = 1000
    episodes = [0, 1, 10, 1000]
    agent = AgentTD(alpha, gamma)
    
    plt.figure()
    
    for i in range(episodes_num + 1):
        if i in episodes:
            plt.plot(agent.V, label=str(i), marker='.')
        s = initial_state
        while s not in end_states:
            next_s = s + np.random.choice((-1, 1))
            r = 0
            if next_s == win_state:
                r = 1
            agent.evaluate_policy(s, r, next_s)
            s = next_s  
        
    plt.plot(true_V, label="true values", marker='.')
    plt.xlabel("State")
    plt.ylabel("Estimated value")
    plt.xlim(1, 5)
    plt.xticks([1, 2, 3, 4, 5], ["A", "B", "C", "D", "E"])
    plt.legend()
    plt.plot()


def mc_plot_errors():
    # Every-visit Monte Carlo Method
    true_V = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
    initial_state = 3
    end_states = (0, 6)
    win_state = 6
    episodes_num = 100
    runs_num = 100
    
    for alpha in (0.01, 0.02, 0.03, 0.04):
        mean_errors = np.zeros(episodes_num)
        for run in range(runs_num):
            agent = AgentMC(alpha)
            for i in range(episodes_num):
                episode_states = []
                s = initial_state
                episode_states.append(s)
                R = 0
                while s not in end_states:
                    s += np.random.choice((-1, 1))
                    episode_states.append(s)
                    if s == win_state:
                        R = 1
                agent.evaluate_policy(episode_states, R)
                mean_error = np.sqrt(np.mean([np.power(agent.V[s] - true_V[s], 2) for s in range(1, 6)]))
                mean_errors[i] += mean_error
        mean_errors /= runs_num
        plt.plot(mean_errors, label="MC, alpha = " + str(alpha), linestyle='--')



def td_plot_errors():
    true_V = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 0]
    initial_state = 3
    end_states = (0, 6)
    win_state = 6
    gamma = 1
    episodes_num = 100
    runs_num = 100
    
    for alpha in (0.05, 0.1, 0.15):
        mean_errors = np.zeros(episodes_num)
        for run in range(runs_num):
            agent = AgentTD(alpha, gamma)
            for i in range(episodes_num):
                s = initial_state
                while s not in end_states:
                    next_s = s + np.random.choice((-1, 1))
                    r = 0
                    if next_s == win_state:
                        r = 1
                    agent.evaluate_policy(s, r, next_s)
                    s = next_s
                mean_error = np.sqrt(np.mean([np.power(agent.V[s] - true_V[s], 2) for s in range(1, 6)]))
                mean_errors[i] += mean_error
        mean_errors /= runs_num
        plt.plot(mean_errors, label="TD, alpha = " + str(alpha))


def plot_errors():
    plt.figure()
    
    mc_plot_errors()
    td_plot_errors()
    
    plt.xlabel("Walks / Episodes")
    plt.ylabel("RMS error, averaged over states")
    plt.legend()
    plt.plot()


def main():
    np.random.seed(42)
    
    td_plot_V()
    plot_errors()
  

if __name__ == '__main__':
    main()