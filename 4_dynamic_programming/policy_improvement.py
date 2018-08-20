import numpy as np


def main():
    theta = 1e-6
    gamma = 1
    reward = -1
    V = np.zeros((4, 4))
    goals = [(0, 0), (3, 3)]
    actions = ['up', 'down', 'right', 'left']
    action_to_index_operation = {'up' : (-1, 0),
                             'down' : (1, 0),
                             'right' : (0, 1),
                             'left' : (0, -1)}
    policy = {}
    for i in range(len(V)):
        for j in range(len(V[i])):
            for a in actions:
                policy[(i, j, a)] = 0.25
    
    delta = theta + 1
    k = 0

    while delta > theta:
        print("k =", k)
        print(V)
        k += 1
        delta = 0
        for i in range(len(V)):
            for j in range(len(V[i])):
                if (i, j) in goals: continue
                old_val = V[i][j]
                new_val = 0
                for a in actions:
                    operation = action_to_index_operation[a]
                    next_i = np.clip(i + operation[0], 0, 3)
                    next_j = np.clip(j + operation[1], 0, 3)
                    new_val += policy[(i, j, a)]*(reward + gamma*V[next_i][next_j])
                V[i][j] = new_val
                delta = max(delta, abs(new_val - old_val))
                

if __name__ == '__main__':
    main()