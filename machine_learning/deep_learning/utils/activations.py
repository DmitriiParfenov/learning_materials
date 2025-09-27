import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def soft_max(z):
    return np.exp(z) / (np.sum(np.exp(z)))

def tanh(z):
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)
