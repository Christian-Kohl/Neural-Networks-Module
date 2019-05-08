import numpy as np


def load_data():
    np.loadtxt('training_data.txt')


print(load_data().shape)
