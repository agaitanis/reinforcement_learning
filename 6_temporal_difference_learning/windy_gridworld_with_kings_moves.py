'''
Reinforcement Learning by Sutton and Barto
6. Temporal-Difference Learning
6.4 Sarsa: On-Policy TD Control
Exercise 6.6: Windy Gridworld with King's Moves
'''
import numpy as np
from collections import defaultdict


def print_greedy_policy(Q, grid_size, action_str):
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            s = (i, j)
            a = np.argmax(Q[s])
            print(action_str[a], end=' ')
        print()
    print()
    
    
def get_next_state(s, a, moves, wind, grid_size):
    return (np.clip(s[0] + moves[a][0] - wind[s[1]], 0, grid_size[0] - 1), 
            np.clip(s[1] + moves[a][1], 0, grid_size[1] - 1))
    

def get_next_action(s, Q, actions, eps):
    if np.random.uniform() < eps:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[s])


def main():
    np.random.seed(0)
    alpha = 0.5
    eps = 0.1
    gamma = 1
    episodes_num = 200
    grid_size = (7, 10)
    start_state = (3, 0)
    end_state = (3, 7)
    wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    moves = ((-1, 0), (1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1))
    action_str = ('u ', 'd ', 'r ', 'l ', 'ur', 'ul', 'dr', 'dl')
    actions = range(len(moves))
    Q = defaultdict(lambda: len(actions)*[0])
    steps = 0
    
    for i in range(episodes_num):
        s = start_state
        a = get_next_action(s, Q, actions, eps)
        while s != end_state:
            new_s = get_next_state(s, a, moves, wind, grid_size)
            r = -1
            new_a = get_next_action(new_s, Q, actions, eps)
            Q[s][a] += alpha*(r + gamma*Q[new_s][new_a] - Q[s][a])
            s, a = new_s, new_a
            steps += 1
            
    print_greedy_policy(Q, grid_size, action_str)

if __name__ == "__main__":
    main()