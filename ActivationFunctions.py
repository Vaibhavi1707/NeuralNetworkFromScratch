import numpy as np
from math import exp

def relu(x):
    return np.array([max(0, x[i]) for i in range(len(x))])

def softmax(x):
    exp_sum = sum(exp(x[i]) for i in range(len(x)))
    return np.array([exp(x[i]) / exp_sum for i in range(len(x))])

def linear(x, m, c):
    return np.array([m * x[i] + c for i in range(len(x))])

def derivative_relu(x):
    return int(x > 0)

def derivative_softmax(x):
    s = softmax(x)
    return [[s[i] * (int(i == j) - s[j]) for j in range(len(x))] for i in range(len(x))]

def derivative_linear(m):
    return m 