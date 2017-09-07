import numpy as np
import matplotlib.pyplot as plt
import math
import random

from Ffnn import Ffnn
from main import util
from util import logistic_sigmoid, normal_distribution


def mydigit():
	n_classes = 10
	nn = Ffnn(49, n_classes, 1, 8)
	m = np.genfromtxt("res/digit-features.txt")
	tmp = np.zeros((m.shape[0], m.shape[1] + 1))
	tmp[:, 0] = np.random.uniform(size=m.shape[0]) > 0.1
	tmp[:, 1:] = m
	m = tmp
	
	def gen_target_vector(index):
		target_vector = np.zeros(n_classes)
		target_vector[index] = 1
		return target_vector
	
	train = m[m[:, 0] == 0][:, 1:]
	test = m[m[:, 0] == 1][:, 1:]
	train_vectors = train[:, :-1]
	test_vectors = test[:, :-1]
	train_targets = np.array(map(gen_target_vector, train[:, -1]))
	test_targets = np.array(map(gen_target_vector, test[:, -1]))
	for round in range(8000):
		if round % 1 == 0:
			total_error = nn.total_error(train_vectors, train_targets)
			print "{}: train = {}/{}, {}; test = {}/{}, {}".format(
				round, nn.count_errors(train_vectors, train_targets), train_vectors.shape[0],
				total_error, nn.count_errors(test_vectors, test_targets),
				test_vectors.shape[0],
				nn.total_error(test_vectors, test_targets))
			outputs = [nn.evaluate(it)[0] for it in test_vectors]
			if total_error < 0.01:
				return
		nn.online_backpropagation_round(train_vectors, train_targets)
	pass


def mymain():
	nn = Ffnn(2, 1, 1, 8)
	m = np.genfromtxt("res/input.txt")
	size = 55
	m = np.random.uniform(size=size * 3).reshape((-1, 3))
	for i in range(size):
		x, y = m[i, :2]
		m[i, 2] = 1 if (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.3 ** 2 else 0
	plt.scatter(m[:, 0], m[:, 1], c=["r" if it == 1 else "b" for it in m[:, 2]])
	plt.show(block=False)
	plt.figure()
	train_vectors = m[:, :-1]
	targets = m[:, -1:]
	print train_vectors
	print targets
	size = 1000
	test_vectors = np.random.uniform(size=size * 2).reshape((-1, 2))
	for round in range(8000):
		if round % 100 == 0:
			print round, nn.total_error(train_vectors, targets)
			outputs = [nn.evaluate(it)[0] for it in test_vectors]
			colors = []
			for i in range(size):
				r = outputs[i]
				color = (2 * r, 0, 1 - 2 * r)
				if r > 0.5:
					color = (0, 1, (r - 0.5) * 2)
				colors.append(color)
			if round > 100:
				plt.close()
			plt.scatter(test_vectors[:, 0], test_vectors[:, 1], c=colors)
			plt.xlim((0, 1))
			plt.ylim((0, 1))
			plt.show(block=False)
		if nn.total_error(train_vectors, targets) < 0.1:
			return
		nn.batch_backpropagation_round(train_vectors, targets)
	print nn
	print "targets:\n{}".format(targets)
	train_results = np.array([nn.evaluate(it) for it in train_vectors])
	print train_results
	print "#errors:", nn.count_errors(train_vectors, targets)
	print "total-error-value:", nn.total_error(train_vectors, targets)
	
	plt.show()
	pass


def myfoo():
	with open("res/test.txt") as myfile:
		data = myfile.read().lower()
	data = data[:1000]
	all_chars = sorted(list(set(data)))
	chars_to_indices = dict(zip(all_chars, range(len(all_chars))))
	data = [chars_to_indices[it] for it in data]
	C = len(all_chars)
	nn = Ffnn(C, C, 1, 15)
	X, T = [], []
	last_char = data[0]
	for c in data[1:]:
		X.append(util.one_1_at(C, last_char))
		T.append(util.one_1_at(C, c))
		last_char = c
	X = np.array(X)
	T = np.array(T)
	for round in range(5):
		if round % 1 == 0:
			print round, nn.count_errors(X, T), nn.total_error(X, T)
		nn.batch_backpropagation_round(X, T)
	print nn
	current_char = 'y'
	res = ""+current_char
	for _ in range(100):
		probs = nn.evaluate(util.one_1_at(C, chars_to_indices[current_char]))
		max_index = np.random.choice(range(C), 1, p=probs)
		next_char = all_chars[max_index]
		res += next_char
		current_char = next_char
	print res
	print X.shape
	pass


if __name__ == "__main__":
	np.random.seed(42)
	random.seed(42)
	# mymain()
	# mydigit()
	myfoo()
	pass
