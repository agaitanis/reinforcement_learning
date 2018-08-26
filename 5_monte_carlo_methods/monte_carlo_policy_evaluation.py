'''
Reinforcement Learning by Sutton and Barto
5. Monte Carlo Methods
5.1 Monte Carlo Policy Evaluation
Example 5.1: Blackjack
'''
import numpy as np


def main():
    episodes_num = 500000
    R = {}
    
    for i in range(episodes_num):
        dealer_showing = min(np.random.randint(1, 14), 10)


if __name__ == '__main__':
    main()