
def sgd_step(target, features, weights, gamma):
	"""One step of weight update for stochastic gradient descent"""
	error = features.dot(weights)-target
	new_weights = weights - gamma * error * features
	return new_weights
