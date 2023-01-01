import math
import random
import numpy as np
from sympy import symbols, integrate


def load_data(n, d):
    l_eta = int(0.01*n*(n-1)/2)
    x = symbols('x')
    f = x ** (-2) * l_eta
    label_vec = np.zeros(d, dtype=int)
    for i in range(1, d + 1):
        label_vec[i - 1] = int(integrate(f, (x, i, i + 1)))
        if label_vec[i - 1] == 0:
            break
    mat_half = np.zeros(int(n * (n - 1) / 2))
    begin = int(n * (n - 1) / 2) - l_eta
    for i in range(len(label_vec)):
        mat_half[begin:begin + label_vec[i]] = i + 1
        begin = begin + label_vec[i]
    np.random.shuffle(mat_half)
    mat = np.zeros(shape=(n, n))
    index = 0
    for i in range(len(mat)):
        for j in range(i):
            mat[i][j] = mat_half[index]
            index += 1
    return mat


def get_factor(d):
    factors = []
    for i in range(2, d):
        if d % i == 0:
            factors.append(i)
    return factors


def get_MSE(list1, list2):
    MSE = 0
    for i in range(len(list1)):
        MSE = MSE + abs(list1[i] - list2[i])**2
    return MSE / len(list1)


def randomized_relation_list(list, eps, d):
    perturbed_list = np.copy(list)
    for i in range(len(list)):
        random_seed = random.random()
        if random_seed > (math.exp(eps) - 1) / (math.exp(eps) + d):
            perturbed_list[i] = random.randrange(0, d + 1, 1)
    return perturbed_list


def relation_frequency_counting_matrix(mat, d):
    rel_fre = np.zeros(d + 1)
    for i in range(len(mat)):
        for j in range(i):
            rel_fre[int(mat[i][j])] += 1
    return rel_fre


def relation_frequency_counting_list(rels, d):
    rel_fre = np.zeros(d + 1)
    for i in range(len(rels)):
        rel_fre[int(rels[i])] += 1
    return rel_fre


def frequency_estimation(noisy_fre, num, eps, d):
    p = math.exp(eps) / (math.exp(eps) + d)
    q = 1 / (math.exp(eps) + d)
    return (noisy_fre - q * num) / (p - q)


def NormMul(n, est_dist):
    estimates = np.copy(est_dist)
    estimates[estimates < 0] = 0
    total = sum(estimates)
    return estimates * n / total


def Fest_R(rels, eps, d):
    noisy_rels = randomized_relation_list(rels, eps, d)
    rel_fre = relation_frequency_counting_list(noisy_rels, d)
    for i in range(len(rel_fre)):
        rel_fre[i] = frequency_estimation(rel_fre[i], len(rels), eps, d)
    rel_fre = NormMul(len(rels), rel_fre)
    if np.sum(rel_fre[1:]) != 0:
        rel_prop = rel_fre[1:] / np.sum(rel_fre[1:])
    else:
        rel_prop = np.zeros(d)
    return rel_fre, rel_prop


def Fest_H(rels, eps, d, k):
    c = int(d / k)
    flash = np.zeros(d + 1, dtype=int)
    for i in range(1, d + 1):
        flash[i] = (i - 1) // k + 1
    hrels = flash[rels]

    hrel_fre, hrel_prop = Fest_R(hrels, eps, c)

    return hrel_fre, hrel_prop, flash


def Fest_C(mat, eps, d, n):
    # randomly dividing matrix
    rels, index = np.zeros(int(n * (n - 1) / 2), dtype=int), 0
    for i in range(n):
        for j in range(i):
            rels[index] = mat[i][j]
            index += 1
    np.random.shuffle(rels)

    rels_R = rels[:int(len(rels)/2)]
    rels_H = rels[int(len(rels)/2):]

    rel_fre, rel_prop = Fest_R(rels_R, eps, d)

    # choosing k
    k = 21.3361 * math.exp(eps) / (math.exp(eps) + 1) + d / (math.exp(eps) + 1)
    factors = get_factor(d)
    k_near = factors[-1]
    for i in range(len(factors)):
        if factors[i] > k:
            break
        k_near = factors[i]

    hrel_fre, hrel_prop, flash = Fest_H(rels_H, eps, d, k_near)

    total_freq = ((np.sum(hrel_fre) - hrel_fre[0]) * n * (n - 1) / 2) / len(rels_H)
    relative_prop = np.zeros(d)
    prop_mat = rel_prop.reshape(int(d/k_near),k_near)
    prop_sum = np.array([np.sum(prop_mat[i]) for i in range(int(d/k_near))])

    for i in range(d):
        if prop_sum[int(i // k_near)] != 0:
            relative_prop[i] = rel_prop[i]*(hrel_prop[int(i // k_near)] / prop_sum[int(i // k_near)])
        else:
            relative_prop[i] = 0.0

    est_freq = total_freq * relative_prop

    return est_freq, k_near


if __name__ == '__main__':
    d = 1000  # size of domain
    n = 100  # number of users
    mat = load_data(n, d)

    for eps in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]:
        epoch = 500
        metr = 0
        k = 0
        for i in range(epoch):
            est_freq, k = Fest_C(mat, eps, d, n)
            real_freq = relation_frequency_counting_matrix(mat, d)
            metr = metr + get_MSE(real_freq[1:], est_freq)
        metr = metr / epoch
        print(eps,"\t", metr, "\t", k)





