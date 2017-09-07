import numpy as np
import math

def logistic_sigmoid(x):
	return 1.0 / (1.0 + np.math.exp(-x))


def logistic_sigmoid_derivative(x):
	z = logistic_sigmoid(x)
	return z * (1.0 - z)

def normal_distribution(mu, sigma_squared, x):
	return 1.0/(math.sqrt(2*math.pi*sigma_squared))*math.exp(-(x-mu)**2/(2*sigma_squared))

def one_1_at(size, index):
	res = np.zeros(size)
	res[index] = 1.0
	return res