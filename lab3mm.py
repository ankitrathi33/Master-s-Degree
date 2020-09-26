import numpy as np
import matplotlib.pyplot as plt

P1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])

P2 = np.array([[0.6, 0.2, 0.1, 0.1],
               [0.4, 0.6, 0, 0],
               [0, 1, 0, 0],
               [0.25, 0.25, 0.25, 0.25]])

init_cond_1 = [0.25, 0.25, 0.25, 0.25]
init_cond_2 = [0.7, 0.1, 0.1, 0.1]
init_cond_3 = [0.2, 0.3, 0.3, 0.2]

def simulate_markov_chain(matrix, init_cond):

    nPi = np.random.multinomial(1, init_cond)
    i = np.where(nPi == 1)[0][0]
    s = list()

    for t in range(20):
        Pi = matrix[i,]
        nPi = np.random.multinomial(1, Pi)
        i = np.where(nPi == 1)[0][0]
        s.append(i)

    s0 = s.count(0)
    s1 = s.count(1)
    s2 = s.count(2)
    s3 = s.count(3)

    return s, s0, s1, s2, s3


def simulate_markov_chains(num_of_iterations, matrix, init_cond):
    for n in range(num_of_iterations):
        s0 = 0
        s1 = 0
        s2 = 0
        s3 = 0

        s, s0_tmp, s1_tmp, s2_tmp, s3_tmp = simulate_markov_chain(matrix, init_cond)

        s0 = s0 + s0_tmp
        s1 = s1 + s1_tmp
        s2 = s2 + s2_tmp
        s3 = s3 + s3_tmp

    sum = s0 + s1 + s2 + s3

    lim_dist = np.array(init_cond).dot(np.linalg.matrix_power(matrix, num_of_iterations))

    print('State 0: ', (s0 * 100) / sum, '%',
          'State 1: ', (s1 * 100) / sum, '%',
          'State 2: ', (s2 * 100) / sum, '%',
          'State 3: ', (s3 * 100) / sum, '%',
          'Limiting distribution: ', lim_dist)


    plt.plot([n for n in range(20)], s)
    plt.show()

print('====== P1 ======')
simulate_markov_chains(100, P1, init_cond_1)
simulate_markov_chains(100, P1, init_cond_2)
simulate_markov_chains(100, P1, init_cond_3)
print('====== P2 ======')
simulate_markov_chains(100, P2, init_cond_1)
simulate_markov_chains(100, P2, init_cond_2)
simulate_markov_chains(100, P2, init_cond_3)

