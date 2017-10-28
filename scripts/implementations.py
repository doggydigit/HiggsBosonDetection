import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

'''
Required implementations:

least_squares_GD(y, tx, initial w,
max iters, gamma)
Linear regression using gradient descent

least_squares_SGD(y, tx, initial w,
max iters, gamma)
Linear regression using stochastic gradient descent

least_squares(y, tx) Least squares regression using normal equations

ridge_regression(y, tx, lambda ) Ridge regression using normal equations

logistic_regression(y, tx, initial w,
max iters, gamma)
Logistic regression using gradient descent or SGD

reg_logistic_regression(y, tx, lambda ,
initial w, max iters, gamma)
Regularized logistic regression using gradient descent
or SGD
'''


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


# Additional funtions to manipulate the data
def white_cubic_features(data, nr_columns, nr_data):
    nr_features=3*nr_columns + 1
    features = np.zeros([nr_data, nr_features])
    for f in range(0, nr_columns):
        features[:, 3*f] = (data[:, f] - np.mean(data[:, f])) / np.std(data[:, f])
        features[:, 3*f+1] = features[:, 3*f]**2
        features[:, 3*f+2] = features[:, 3*f]**3
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    a = np.ones(x.shape[0])
    for deg in np.arange(1, degree+1):
        b = np.power(x, deg)
        b[b == (-999)**deg] = -999
        a = np.c_[a, b]       
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

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    indicies = np.random.permutation(x.shape[0])
    index_number = int(ratio*x.shape[0])
    return (x[indicies[:index_number], ], x[indicies[index_number:] ,], y[indicies[:index_number], ], y[indicies[index_number:] ,])

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x

def accuracy(weights, features, targets, nr_traindata, model_type):
    if(model_type == "linear"):        
        train_predictions = predict_labels(weights, features)
    elif(model_type == "logistic"):
         train_predictions = predict_labels_lg(weights, features)
    return 1-(nr_traindata-train_predictions.dot(targets))/(2*nr_traindata)

def cross_validation(y, x, k_indices, k, lambda_, degree, model_type, max_iters = 1000, gamma = 0.01):
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
        x_train, _, _ = standardize(x_train)
        x_test = build_poly(x_test, degree)
        x_test, _, _ = standardize(x_test)
        
        initial_w = np.random.random(x_train.shape[1])
        
        if(model_type == "linear"):
            weights = ridge_regression(y_train, x_train, lambda_)
        elif(model_type == "logistic"):
             weights, _ = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
        #loss_tr_arr[i] = np.sqrt(2 * compute_mse(y_train, x_train, weights))
        loss_tr_arr[i] = accuracy(weights, x_train, y_train, x_train.shape[0], model_type)
       # loss_te_arr[i] = np.sqrt(2 * compute_mse(y_test, x_test, weights))
        loss_te_arr[i] = accuracy(weights, x_test, y_test, x_test.shape[0], model_type)
        
    loss_tr = np.mean(loss_tr_arr)
    loss_te = np.mean(loss_te_arr)
    return loss_tr, loss_te

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()

def plot_corr_matrix(corr_matrix, labels):
    fig_cor, axes_cor = plt.subplots(1,1)
    fig_cor.set_size_inches(12, 12)

    myimage = axes_cor.imshow(corr_matrix, cmap='seismic', interpolation='nearest', vmax=1, vmin = -1)

    plt.colorbar(myimage)

    axes_cor.set_xticks(np.arange(0,corr_matrix.shape[0], corr_matrix.shape[0]*1.0/len(labels)))
    axes_cor.set_yticks(np.arange(0,corr_matrix.shape[1], corr_matrix.shape[1]*1.0/len(labels)))

    axes_cor.set_xticklabels(labels)
    axes_cor.set_yticklabels(labels)
    plt.xticks(rotation=90)

    plt.draw()
    
def insert_mean_for_nan(data):
    for i in range(data.shape[1]):
        data[data[:, i] == -999, i] = np.nan
        data[np.isnan(data[:, i]), i] = np.nanmean(data[:, i])
    
# Functions for linear regression task

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

def compute_loss(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_gradient(y, tx, w, lambda_ = 0):
    """Compute the gradient."""
    error = y - tx.dot(w)
    return (-1/y.shape[0]) * np.dot(tx.T, error.T) + 2*lambda_*w

def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_ = 0):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        w = w - gamma*compute_gradient(y, tx, w, lambda_)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def compute_stoch_gradient(y, tx, w, lambda_):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w, lambda_)

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, lambda_ = 0):
    """Stochastic gradient descent algorithm for linear regression."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):              
        n = np.random.random_integers(size = batch_size, low = 0, high = y.shape[0] - 1)
        w = w - gamma*compute_stoch_gradient(y[n], tx[n], w, lambda_)
        ws.append(w)
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        print("SGD({bi}/{ti}): loss={l}, norm of weights={w}, gamma={g}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w = w.dot(w), g = gamma))

    return losses, ws

# Functions for logistic regression

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w))

def calculate_gradient(y, tx, w, lambda_ = 0):
    """compute the gradient of loss."""
    return tx.T.dot((sigmoid(tx.dot(w)) - y)) + lambda_ * np.abs(w)

def learning_by_gradient_descent(y, tx, w, gamma, lambda_ = 0):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w, lambda_ = 0)
    w = w - gamma*grad
    return loss, w
         
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    weights = initial_w
    batch_size = 1
    for i in range(max_iters):
        n = np.random.random_integers(size = batch_size, low = 0, high = y.shape[0] - 1)
        weigths = weights - gamma * calculate_gradient(y[n], tx[n], weights, lambda_)
        #print("Weights = " + str(weights))
    loss = calculate_loss(y, tx, weights)
    return weights, loss
         
def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    temp = tx.dot(w)
    S = np.identity(len(tx)) * sigmoid(temp) * (1 - sigmoid(temp))
    H = tx.T.dot(S).dot(tx)
    return H

def logistic_regression_newton(y, tx, w):
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
    loss, gradient, hessian = logistic_regression_newton(y, tx, w)
    w = w - np.linalg.inv(hessian).dot(gradient)
    return loss, w

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss, gradient, hessian = logistic_regression_newton(y, tx, w)
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
