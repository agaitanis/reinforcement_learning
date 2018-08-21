import numpy as np

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3


def get_next_s(s, a):
    if a == UP:
        if s >= 4: return s - 4
    elif a == DOWN:
        if s < 12: return s + 4
    elif a == RIGHT:
        if (s + 1)%4 != 0: return s + 1
    elif a == LEFT:
        if s%4 != 0: return s - 1
    return s


def main():
    theta = 1e-6
    actions_num = 4
    states_num = 16
    gamma = 1
    V = np.zeros(states_num)
    policy = np.zeros((states_num, actions_num))
    P = np.zeros((states_num, actions_num, states_num))
    R = np.full((states_num, actions_num, states_num), -1)

    for s in range(1, states_num - 1):
        for a in range(actions_num):
            policy[s, a] = 0.25
    
    for s in range(1, states_num - 1):
        for a in range(actions_num):
            next_s = get_next_s(s, a)
            P[s, a, next_s] = 1

    delta = theta + 1
    k = 0

    while delta > theta:
        print("k =", k)
        print(V.reshape(4, 4))
        k += 1
        delta = 0
        for s in range(states_num):
            old_v = V[s]
            new_v = 0
            for a in range(actions_num):
                temp_sum = 0
                for next_s in range(states_num):
                    temp_sum += P[s, a, next_s]*(R[s, a, next_s] + gamma*V[next_s])
                new_v += policy[s, a]*temp_sum
            V[s] = new_v
            delta = max(delta, abs(new_v - old_v))


if __name__ == '__main__':
    main()