'''
Reinforcement Learning by Sutton and Barto
6. Temporal-Difference Learning
6.4 Sarsa: On-Policy TD Control
Example 6.5: Windy Gridworld
'''
import numpy as np


UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


def main():
    eps = 0.1
    alpha = 0.1
    episodes_num = 170
    Q = np.zeros((7, 10, 4))
    start_state = (3, 0)
    end_state = (3, 7)
    
    for i in range(episodes_num):
        pass
    

if __name__ == "__main__":
    main()