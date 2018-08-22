import numpy as np

states_num1 = 21
states_num2 = 21
actions_num = 11

def get_possible_actions(s1, s2):
    pass

def P(s1, s2, a, next_s1, next_s2):
    pass

def R(s1, s2, a, next_s1, next_s2):
    pass

def action_to_index(a):
    return a + 5

def main():
    theta = 1e-6
    gamma = 1
    V = np.zeros(states_num1, states_num2)
    policy = np.zeros(states_num1, states_num2, actions_num)
    delta = theta + 1
    k = 0
    while delta > theta:
        print("k =", k)
        print(V)
        k += 1
        delta = 0
        for s1 in range(states_num1):
            for s2 in range(states_num2):
                old_v = V[s1, s2]
                new_v = 0
                actions = get_possible_actions(s1, s2)
                for a in actions:
                    temp_sum = 0
                    for next_s1 in range(states_num1):
                        for next_s2 in range(states_num2):
                            temp_sum += P(s1, s2, a, next_s1, next_s2)
                            *(R(s1, s2, a, next_s1, next_s2)
                            + gamma*V[next_s1, next_s2])
                    a_index = action_to_index(a)
                    new_v += policy[s1, s2, a_index]*temp_sum
                V[s1, s2] = new_v
                delta = max(delta, abs(new_v - old_v))


if __name__ == '__main__':
    main()