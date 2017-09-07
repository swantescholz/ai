import numpy as np

from util import logistic_sigmoid


class Ffnn():
	LEARNING_RATE = 0.5
	
	def __init__(self, n_inputs, n_outputs, n_hidden_layers=1, hidden_layer_size=None):
		if hidden_layer_size is None:
			hidden_layer_size = (n_inputs + n_outputs) / 2
		self.n_inputs = n_inputs
		self.n_outputs = n_outputs
		self.n_hidden_layers = n_hidden_layers
		self.hidden_layer_size = hidden_layer_size
		self.ws = []
		if n_hidden_layers == 0:
			self.ws.append(np.random.uniform(size=(n_outputs, n_inputs + 1)))
		else:
			self.ws.append(np.random.uniform(size=(hidden_layer_size, n_inputs + 1)))
			for _ in range(n_hidden_layers - 1):
				self.ws.append(np.random.uniform(size=(hidden_layer_size, hidden_layer_size + 1)))
			self.ws.append(np.random.uniform(size=(n_outputs, hidden_layer_size + 1)))
	
	def __str__(self):
		return "FFNN weight matrices:\n" + "\n".join(str(w) for w in self.ws)
	
	def evaluate(self, input_vector):
		assert input_vector.shape[0] == self.n_inputs
		x = np.ones(self.n_inputs + 1)
		x[:-1] = input_vector
		for i, w in enumerate(self.ws):
			tmp = w.dot(x)
			x = np.ones(tmp.shape[0] + 1)
			if i < len(self.ws) - 1:
				tmp = map(logistic_sigmoid, tmp)
			else:
				tmp = np.exp(tmp)
				tmp /= np.sum(tmp)
			x[:-1] = tmp
		return x[:-1]
	
	# targets: desired outputs (using cross-entropy error function now)
	def total_error(self, input_vectors, targets):
		def error_of(y, t):
			return -np.sum(ti * np.log(yi) for (yi, ti) in zip(y, t))
		
		return 1.0 / input_vectors.shape[0] * np.sum(error_of(self.evaluate(x), t) for (x, t)
		                                             in zip(input_vectors, targets))
	
	def get_layer_size(self, layer_index):
		assert layer_index >= 0 and layer_index <= self.n_hidden_layers + 1
		if layer_index == 0:
			return self.n_inputs
		if layer_index == self.n_hidden_layers + 1:
			return self.n_outputs
		return self.hidden_layer_size
	
	# returns list of matrices of weight deltas
	def _compute_weight_deltas_for_backpropagation_step(self, input_vector, target):
		assert input_vector.shape[0] == self.n_inputs
		assert target.shape[0] == self.n_outputs
		x = np.ones(self.n_inputs + 1)
		x[:-1] = input_vector
		os = [x]
		for w in self.ws:
			tmp = w.dot(x)
			x = np.ones(tmp.shape[0] + 1)
			tmp = np.array(map(logistic_sigmoid, tmp))
			x[:-1] = tmp
			os.append(x)
		delta_factors = [0] * len(self.ws)
		d = np.zeros(self.n_outputs)
		for j in range(self.n_outputs):
			oj = os[-1][j]
			d[j] = (oj - target[j]) * oj * (1.0 - oj)
		delta_factors[-1] = d
		for layer_index in reversed(range(self.n_hidden_layers)):
			d = np.zeros(self.get_layer_size(layer_index + 1))
			for j in range(d.shape[0]):
				oj = os[layer_index + 1][j]
				dj = 0.0
				for l in range(delta_factors[layer_index + 1].shape[0]):
					dj += delta_factors[layer_index + 1][l] * self.ws[layer_index + 1][l, j]
				dj *= oj * (1.0 - oj)
				d[j] = dj
			delta_factors[layer_index] = d
		weight_deltas = []
		for layer_index in range(len(self.ws)):
			w = np.zeros(self.ws[layer_index].shape)
			for j in range(w.shape[1]):
				for l in range(w.shape[0]):
					w[l, j] = - Ffnn.LEARNING_RATE * delta_factors[layer_index][l] * os[layer_index][j]
			weight_deltas.append(w)
		return weight_deltas
	
	def _apply_deltas_to_weights(self, weight_deltas):
		for i in range(len(self.ws)):
			w = self.ws[i]
			for j in range(w.shape[1]):
				for l in range(w.shape[0]):
					w[l, j] = w[l, j] + weight_deltas[i][l, j]
	
	def online_backpropagation_round(self, training_vectors, targets):
		for (x, t) in zip(training_vectors, targets):
			weight_deltas = self._compute_weight_deltas_for_backpropagation_step(x, t)
			self._apply_deltas_to_weights(weight_deltas)
	
	def batch_backpropagation_round(self, training_vectors, targets):
		weight_deltas = None
		for (x, t) in zip(training_vectors, targets):
			new_weight_deltas = self._compute_weight_deltas_for_backpropagation_step(x, t)
			if weight_deltas is None:
				weight_deltas = new_weight_deltas
			else:
				for i in range(len(new_weight_deltas)):
					weight_deltas[i] += new_weight_deltas[i]
		self._apply_deltas_to_weights(weight_deltas)
	
	def count_errors(self, input_vectors, targets):
		outputs = np.array([self.evaluate(it) for it in input_vectors])
		if self.n_outputs == 1:
			return np.sum(np.absolute(outputs - targets) > 0.5)
		n_correct = 0
		for o, t in zip(outputs, targets):
			best_index = max(range(self.n_outputs), key=lambda it: o[it])
			if t[best_index] == 1.0:
				n_correct += 1
		return targets.shape[0] - n_correct
