import numpy as np

def white_cubic_features(data, nr_columns, nr_data):
	nr_features=3*nr_columns + 1
	features = np.zeros([nr_data,nr_features])
	for f in range(0,nr_columns):
		features[:,3*f] = (data[:,f] - np.mean(data[:,f])) / np.std(data[:,f])
		features[:,3*f+1] = features[:,3*f]**2
		features[:,3*f+2] = features[:,3*f]**3
	features[:,nr_features-1] = np.ones([nr_data,1])[:,0] 
	return features

def sgd_step(target, features, weights, gamma):
	"""One step of weight update for stochastic gradient descent"""
	error = features.dot(weights)-target
	new_weights = weights - gamma * error * features
	return new_weights
