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

def ridge_regression(y, tx, lambda_):
    """Returns only weights"""
    shape = np.shape(np.dot(tx.T,tx))
    a = np.dot(tx.T,tx) + lambda_*(2.0*len(y)) * np.identity(shape[0])
    b = np.dot(tx.T,y)
    return np.linalg.solve(a,b)

def least_squares(y, tx):
    a = np.dot(tx.T,tx)
    b = np.dot(tx.T,y)
    return np.linalg.solve(a,b)

def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    a = np.ones(x.shape)
    for deg in np.arange(1, degree+1):
        a = np.c_[a, np.power(x, deg)]       
    return a

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    loss_tr_arr = np.zeros(k)
    loss_te_arr = np.zeros(k)
    for i in range(k):
        mask = np.zeros(y.shape[0], dtype=bool)
        mask[k_indices[i]] = True
        x_train = x[~mask]
        y_train = y[~mask]
        x_test = x[mask]
        y_test = y[mask]

        x_train = build_poly(x_train, degree)
        x_test = build_poly(x_test, degree)
        
        weights = ridge_regression(y_train, x_train, lambda_)
        
        loss_tr_arr[i] = np.sqrt(2 * compute_mse(y_train, x_train, weights))
        loss_te_arr[i] = np.sqrt(2 * compute_mse(y_test, x_test, weights))
        
    loss_tr = np.mean(loss_tr_arr)
    loss_te = np.mean(loss_te_arr)
    return loss_tr, loss_te

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    indicies = np.random.permutation(x.shape[0])
    index_number = int(ratio*x.shape[0])
    return (x[indicies[:index_number], ], x[indicies[index_number:] ,], y[indicies[:index_number], ], y[indicies[index_number:] ,])

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T.dot((sigmoid(tx.dot(w)) - y))

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma*grad
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    temp = tx.dot(w)
    S = np.identity(len(tx)) * sigmoid(temp) * (1 - sigmoid(temp))
    H = tx.T.dot(S).dot(tx)
    return H

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    return loss, gradient, hessian

def learning_by_newton_method(y, tx, w):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression(y, tx, w)
    w = w - np.linalg.inv(hessian).dot(gradient)
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss, gradient, hessian = logistic_regression(y, tx, w)
    reg = (lambda_/2) * w.T.dot(w)
    return loss+reg, gradient, hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*np.linalg.inv(hessian).dot(gradient)
    return loss, w

