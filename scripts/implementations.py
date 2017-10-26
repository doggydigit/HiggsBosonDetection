import numpy as np


def second_order_features(data, nr_columns, nr_data):
    nr_features= nr_columns**2 + nr_columns + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    # Whitening features
    for f in range(0, nr_features-1):
        features[:, f] = (features[:, f] - np.mean(features[:, f])) / np.std(features[:, f])

    # Add bias
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features


def white_cubic_features(data, nr_columns, nr_data):
    nr_features=3*nr_columns + 1
    features = np.zeros([nr_data, nr_features])
    for f in range(0, nr_columns):
        features[:, 3*f] = (data[:, f] - np.mean(data[:, f])) / np.std(data[:, f])
        features[:, 3*f+1] = features[:, 3*f]**2
        features[:, 3*f+2] = features[:, 3*f]**3
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features


def sgd_step(target, features, weights, gamma):
    """One step of weight update for stochastic gradient descent"""
    error = features.dot(weights)-target
    new_weights = weights - gamma * error * features
    return new_weights


def ridge_regression(y, tx, lambda_):
    """Returns only weights"""
    shape = np.shape(np.dot(tx.T, tx))
    a = np.dot(tx.T, tx) + lambda_*(2.0*len(y)) * np.identity(shape[0])
    b = np.dot(tx.T, y)
    return np.linalg.solve(a, b)


def least_squares(y, tx):
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    return np.linalg.solve(a, b)


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse
