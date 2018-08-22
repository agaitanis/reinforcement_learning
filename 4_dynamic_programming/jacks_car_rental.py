import numpy as np

states_num1 = 21
states_num2 = 21


def P(s1, s2, a, next_s1, next_s2):
    pass

def R(s1, s2, a, next_s1, next_s2):
    pass


def main():
    theta = 1e-6
    gamma = 0.9
#    actions = range(-5, 6)
    V = np.zeros(states_num1, states_num2)
    policy = np.zeros(states_num1, states_num2)
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
                a = policy(s1, s2)
                for next_s1 in range(states_num1):
                    for next_s2 in range(states_num2):
                        new_v += P(s1, s2, a, next_s1, next_s2)
                        *(R(s1, s2, a, next_s1, next_s2)
                        + gamma*V[next_s1, next_s2])
                V[s1, s2] = new_v
                delta = max(delta, abs(new_v - old_v))


if __name__ == '__main__':
    main()
