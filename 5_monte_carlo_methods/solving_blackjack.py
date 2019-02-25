'''
Reinforcement Learning by Sutton and Barto
5. Monte Carlo Methods
5.3 Monte Carlo Control
Example 5.3: Solving Blackjack
'''
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


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
    actions = []
    player_sum = np.random.randint(11, 22)
    player_cards_num = np.random.randint(2, 4)
    usable_ace = np.random.uniform() > 0.5
    action = np.random.randint(0, 2)
    s = (player_sum, dealer_showing, usable_ace)
    states.append(s)
    actions.append(action)
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
        actions.append(action)
    
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
    return reward, states, actions


def get_initial_policy():
    policy = defaultdict(lambda: HIT)
    for player_sum in range(20, 22):
        for dealer_showing in range(1, 11):
            for usable_ace in (True, False):
                s = (player_sum, dealer_showing, usable_ace)              
                policy[s] = STICK
    return policy


def plot_policy(policy, usable_ace, ax):
    policy_arr = []
    for player_sum in range(22):
        policy_arr.append([])
        for dealer_showing in range(11):
            s = (player_sum, dealer_showing, usable_ace)
            policy_arr[-1].append(policy[s])
    ax.matshow(policy_arr, cmap=cm.coolwarm)
    ax.xaxis.set_ticks_position('bottom')
    ax.invert_yaxis()
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(10.5, 21.5)


def plot_V(V, usable_ace, ax):
    z = np.zeros((10, 10))
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            z[player_sum - 12][dealer_showing - 1] = V[player_sum, dealer_showing, usable_ace]
        
    x = range(1, 11)
    y = range(12, 22)
    x, y = np.meshgrid(x, y)
    
    ax.plot_surface(x, y, z, cmap=cm.coolwarm)
    ax.set_xlim(1, 11)
    ax.set_ylim(11, 21)
    ax.set_zlim(-1, 1)


def plot_policy_and_V(policy, V):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(221)
    ax1.set_title('π*\nUsable Ace')
    plot_policy(policy, True, ax1)
    ax2 = fig.add_subplot(223)
    ax2.set_title('No Usable Ace')
    ax2.set_xlabel('Dealer Showing')
    ax2.set_ylabel('Player Sum')
    plot_policy(policy, False, ax2)
    
    ax3 = fig.add_subplot(222, projection='3d')
    ax3.set_title('V*\nUsable Ace')
    plot_V(V, True, ax3)
    ax4 = fig.add_subplot(224, projection='3d')
    ax4.set_title('No Usable Ace')
    ax4.set_xlabel('Dealer Showing')
    ax4.set_ylabel('Player Sum')
    plot_V(V, False, ax4)
    
    plt.subplots_adjust(hspace=0.5)
    plt.show()


class Agent(object):
    def __init__(self):
        self.Q = defaultdict(lambda: [0, 0])
        self.cnt_Q = defaultdict(lambda: [0, 0])
        self.V = defaultdict(float)
        self.cnt_V = defaultdict(int)
        self.policy = defaultdict(int)

    def evaluate_policy(self, reward, states, actions):
        for s, a in zip(states, actions):
            self.cnt_Q[s][a] += 1
            self.Q[s][a] += (reward - self.Q[s][a])/self.cnt_Q[s][a]
        for s in states:
            self.cnt_V[s] += 1
            self.V[s] += (reward - self.V[s])/self.cnt_V[s]
    
    def improve_policy(self, states):
        for s in states:
            self.policy[s] = np.argmax(self.Q[s])


def main():
    np.random.seed(42)
    episodes_num = 2000000
    agent = Agent()
    
    for i in range(episodes_num):
        reward, states, actions = generate_episode(agent.policy)
        agent.evaluate_policy(reward, states, actions)
        agent.improve_policy(states)

    plot_policy_and_V(agent.policy, agent.V)


if __name__ == '__main__':
    main()