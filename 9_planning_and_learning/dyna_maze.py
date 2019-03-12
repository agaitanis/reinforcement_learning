# -*- coding: utf-8 -*-
"""
Reinforcement Learning by Sutton and Barto
9. Planning and Learning
9.2: Integrating Planning, Acting and Learning
Example 9.1 Dyna Maze (Dyna-Q algorithm)
"""
import numpy as np
from collections import defaultdict
from enum import Enum
import matplotlib.pyplot as plt


class Square(Enum):
    FREE = 0
    OBSTACLE = 1


class Environment(object):
    def __init__(self):
        self.grid = np.full((6, 9), Square.FREE)
        self.start_state = (2, 0)
        self.goal_state = (0, 8)
        self.grid[1:4, 2] = Square.OBSTACLE
        self.grid[0:3, 7] = Square.OBSTACLE
        self.grid[4, 5] = Square.OBSTACLE
    
    
    def respond_to_move(self, s, move):
        new_s = tuple(np.array(s) + np.array(move))
        
        if new_s[0] < 0: return 0.0, s
        if new_s[1] < 0: return 0.0, s
        if new_s[0] >= self.grid.shape[0]: return 0.0, s
        if new_s[1] >= self.grid.shape[1]: return 0.0, s
        if self.grid[new_s] is Square.OBSTACLE: return 0.0, s
        if new_s == self.goal_state: return 1.0, new_s
        
        return 0.0, new_s


class Agent(object):
    def __init__(self, planning_steps):
        self.alpha = 0.1
        self.gamma = 0.95
        self.eps = 0.1
        self.planning_steps = planning_steps
        self.action_to_move = ((-1, 0), (1, 0), (0, 1), (0, -1))
        self.action_to_str = ('^', 'v', '>', '<')
        self.actions = range(len(self.action_to_move))
        self.Q = defaultdict(lambda: np.zeros(len(self.actions)))
        self.Model = {}
    
    
    def select_action(self, s):
        if np.random.uniform() < self.eps:
            return np.random.choice(self.actions)
        else:
            q = self.Q[s]
            return np.random.choice(np.flatnonzero(q == max(q)))
    
    
    def plan_policy(self):
        visited_states = list(self.Model.items())
        
        for i in range(self.planning_steps):
            j = np.random.randint(len(visited_states))
            (s, a), (r, new_s) = visited_states[j]
            self.Q[s][a] += self.alpha*(r + self.gamma*max(self.Q[new_s]) - self.Q[s][a])


    def learn_policy(self, s, a, r, new_s):
        self.Q[s][a] += self.alpha*(r + self.gamma*max(self.Q[new_s]) - self.Q[s][a])
        self.Model[s, a] = (r, new_s)


    def improve_policy(self, s, a, r, new_s):
        self.learn_policy(s, a, r, new_s)
        self.plan_policy()


def run_maze_experiment(planning_steps):
    np.random.seed(0)
    env = Environment()
    runs_num = 30
    episodes_num = 50
    steps = np.zeros(episodes_num)
    episodes = range(episodes_num)
    
    print('planning steps =', planning_steps)
    
    for i in range(runs_num):
        agent = Agent(planning_steps)
        for j in episodes:
            steps_num = 0
            s = env.start_state
            while s != env.goal_state:
                a = agent.select_action(s)
                move = agent.action_to_move[a]
                r, new_s = env.respond_to_move(s, move)
                agent.improve_policy(s, a, r, new_s)
                steps_num += 1
                s = new_s
            steps[j] += steps_num
    
    steps /= runs_num
    
    plt.plot(episodes, steps, label=str(planning_steps) + " planning steps")


def main():
    plt.figure()
        
    run_maze_experiment(0)
    run_maze_experiment(5)
    run_maze_experiment(50)
    
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episodes')
    plt.ylim(0, 800)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()