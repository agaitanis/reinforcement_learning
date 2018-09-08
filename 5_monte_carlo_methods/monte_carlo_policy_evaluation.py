'''
Reinforcement Learning by Sutton and Barto
5. Monte Carlo Methods
5.1 Monte Carlo Policy Evaluation
Example 5.1: Blackjack
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


HIT = 0
STICK = 1


def generate_episode(policy):
    dealer_showing = min(np.random.randint(1, 14), 10)
    dealer_hidden = min(np.random.randint(1, 14), 10)
    dealer_sum = 0
    dealer_usable_ace = False
    dealer_sum += dealer_showing
    if dealer_showing == 1:
        dealer_sum += 10
        dealer_usable_ace = True
    dealer_sum += dealer_hidden
    if dealer_sum > 21 and dealer_usable_ace:
        dealer_sum -= 10
        dealer_usable_ace = False
    if dealer_hidden == 1 and dealer_sum <= 11:
        dealer_sum += 10
        dealer_usable_ace = True
        
    states = []
    player_sum = 0
    player_cards_num = 0
    usable_ace = False
    action = HIT
    while action == HIT:
        player_card = min(np.random.randint(1, 14), 10)
        player_cards_num += 1
        player_sum += player_card
        if player_sum > 21 and usable_ace:
            player_sum -= 10
            usable_ace = False
        if player_card == 1 and player_sum <= 11:
            player_sum += 10
            usable_ace = True
        if player_sum <= 11:
            continue
        if player_sum > 21:
            break
        s = (player_sum, dealer_showing, usable_ace)
        action = policy[s]
        states.append(s)
        
    reward = 0
    if player_sum == 21 and player_cards_num == 2:
        if dealer_sum == 21:
            reward = 0
        else:
            reward = 1
    elif player_sum > 21:
        reward = -1
    else:
        while dealer_sum < 17:
            dealer_card = min(np.random.randint(1, 14), 10)
            dealer_sum += dealer_card
            if dealer_sum > 21 and dealer_usable_ace:
                dealer_sum -= 10
                dealer_usable_ace = False
            if dealer_card == 1 and dealer_sum <= 11:
                dealer_sum += 10
                dealer_usable_ace = True
        if dealer_sum > 21:
            reward = 1
        else:
            if player_sum > dealer_sum:
                reward = 1
            elif player_sum < dealer_sum:
                reward = -1
            else:
                reward = 0
    return reward, states


def plot_V(V, ax1, ax2):
    z1 = np.zeros((10, 10))
    z2 = np.zeros((10, 10))
    
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            z1[player_sum - 12][dealer_showing - 1] = V[player_sum, dealer_showing, True]
            z2[player_sum - 12][dealer_showing - 1] = V[player_sum, dealer_showing, False]

    x = range(1, 11)
    y = range(12, 22)
    x, y = np.meshgrid(x, y)
    
    ax1.plot_surface(x, y, z1, cmap=cm.coolwarm)
    ax1.set_xlim(1, 10)
    ax1.set_ylim(12, 21)
    ax1.set_zlim(-1, 1)
    
    ax2.plot_surface(x, y, z2, cmap=cm.coolwarm)
    ax2.set_xlim(1, 10)
    ax2.set_ylim(12, 21)
    ax2.set_zlim(-1, 1)


def policy_evaluation(episodes_num, ax1, ax2):
    np.random.seed(0)
    V = {}
    cnt = {}
    policy = {}
    
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            for usable_ace in (True, False):
                s = (player_sum, dealer_showing, usable_ace)
                V[s] = 0
                cnt[s] = 0
                if player_sum >= 20:
                    policy[s] = STICK
                else:
                    policy[s] = HIT
    
    for i in range(episodes_num):
        reward, states = generate_episode(policy)
        for s in states:
            cnt[s] += 1
            V[s] += (reward - V[s])/cnt[s]

    plot_V(V, ax1, ax2)


def main():
    fig = plt.figure()
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title('After 10,000 episodes\nUsable ace')
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    ax2.set_title('No usable ace')
    policy_evaluation(10000, ax1, ax2)
    
    ax1 = fig.add_subplot(2, 2, 2, projection='3d')
    ax1.set_title('After 500,000 episodes\nUsable ace')
    ax2 = fig.add_subplot(2, 2, 4, projection='3d')
    ax2.set_title('No usable ace')
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    policy_evaluation(500000, ax1, ax2)
    
    plt.show()


if __name__ == '__main__':
    main()