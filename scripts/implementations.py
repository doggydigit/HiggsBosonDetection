import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
import proj1_helpers
import warnings

'''
Required implementations:

least_squares_GD(y, tx, initial w,
max iters, gamma)
Linear regression using gradient descent

least_squares_SGD(y, tx, initial w,
max iters, gamma)
Linear regression using stochastic gradient descent

DONE: least_squares(y, tx) Least squares regression using normal equations

DONE: ridge_regression(y, tx, lambda ) Ridge regression using normal equations

DONE: logistic_regression(y, tx, initial w,
max iters, gamma)
Logistic regression using gradient descent or SGD

DONE: reg_logistic_regression(y, tx, lambda ,
initial w, max iters, gamma)
Regularized logistic regression using gradient descent
or SGD
'''


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression model using gradient descent algorithm
    
    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.
        
        initial_w : np.array
        Initial weights/starting point for optimization.
        
        max_iters : int
        Maximal number of iterations for algorithm.
        
        gamma : float
        Learning rate of the gradient descent.

    Returns:
        (weights, loss) : (np.array, float) 
        
        weights : Final weights for the model.
        loss : Final loss value.     
    """
    losses, weights = gradient_descent(y, tx, initial_w, max_iters, gamma, 0)
    return weights[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Linear regression model using stochastic gradient descent algorithm
    
    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.
        
        initial_w : np.array
        Initial weights/starting point for optimization.
        
        max_iters : int
        Maximal number of iterations for algorithm.
        
        gamma : float
        Learning rate of the gradient descent.

    Returns:
        (weights, loss) : (np.array, float) 
        weights : Final weights for the model.
        
        loss : Final loss value.     
    """
    losses, weights = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma, 0)
    return weights[-1], losses[-1]


def add_mass_binary(data):
    nrdata, nrcolumns = data.shape
    newdata = np.zeros((nrdata, nrcolumns+1))
    mask = data[:, 0] == -999
    newdata[:, 0:nrcolumns] = data
    newdata[:, -1] = mask.astype(int)
    return newdata


def split_data_by_jet_num(data, testing=True, labels=0):
    jet_num_index = 22
    mask0 = [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    mask1 = [4, 5, 6, 12, 22, 26, 27, 28]
    mask2 = [22]
    mask3 = [22]
    nr_columns = len(data[0])
    jetmask0 = np.ones(nr_columns, dtype=bool)
    jetmask1 = np.ones(nr_columns, dtype=bool)
    jetmask2 = np.ones(nr_columns, dtype=bool)
    jetmask3 = np.ones(nr_columns, dtype=bool)
    jetmask0[mask0] = False
    jetmask1[mask1] = False
    jetmask2[mask2] = False
    jetmask3[mask3] = False
    m0 = np.ndarray.tolist(np.where(data[:, jet_num_index] == 0)[0])
    m1 = np.ndarray.tolist(np.where(data[:, jet_num_index] == 1)[0])
    m2 = np.ndarray.tolist(np.where(data[:, jet_num_index] == 2)[0])
    m3 = np.ndarray.tolist(np.where(data[:, jet_num_index] == 3)[0])
    splitdata0 = data[m0, :]
    splitdata0 = splitdata0[:, np.ndarray.tolist(np.where(jetmask0)[0])]
    splitdata1 = data[m1, :]
    splitdata1 = splitdata1[:, np.ndarray.tolist(np.where(jetmask1)[0])]
    splitdata2 = data[m2, :]
    splitdata2 = splitdata2[:, np.ndarray.tolist(np.where(jetmask2)[0])]
    splitdata3 = data[m3, :]
    splitdata3 = splitdata3[:, np.ndarray.tolist(np.where(jetmask3)[0])]

    if not testing:
        m0 = labels[m0]
        m1 = labels[m1]
        m2 = labels[m2]
        m3 = labels[m3]

    return splitdata0, splitdata1, splitdata2, splitdata3, m0, m1, m2, m3


def build_features(data, order=2, testing=False, means=0, stds=0):

    # Getting the indexes of features that are non-negative
    defpos_nr = 0
    defpos_indexes = []
    for i in range(0, len(data[0])):
        if np.min(data[:, i]) > -0.0000001:
            defpos_nr += 1
            defpos_indexes = defpos_indexes + [i]

    # Some variable initializations
    nr_data, nr_columns = data.shape
    nr_features = nr_columns**2 + order*nr_columns + 2*defpos_nr + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms including interaction terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    # polynomial terms
    for o in range(3, order + 1):
        for f in range(0, nr_columns):
            features[:, nr_columns ** 2 + (o-2)*nr_columns + f] = data[:, f] ** o

    # cubic root terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + (order-1) * nr_columns + f] = np.cbrt(data[:, f])

    # log terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + order * nr_columns + f] = np.log(data[:, defpos_indexes[f]] + 1)

    # square root terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + order * nr_columns + defpos_nr + f] = np.sqrt(data[:, defpos_indexes[f]])

    # Making ure the means and standard deviations of the training set are used for whitening
    if not testing:
        means = np.zeros(nr_features)
        stds = np.zeros(nr_features)
        for f in range(0, nr_features-1):
            means[f] = np.mean(features[:, f])
            stds[f] = np.std(features[:, f])

    # Whitening features
    for f in range(0, nr_features - 1):
        features[:, f] = (features[:, f] - means[f]) / stds[f]

    # Add bias
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features, means, stds


def build_test_features(data, order=2, testing=False, means=0, stds=0):
    defpos_nr = 0
    defpos_indexes = []
    for i in range(0, len(data[0])):
        if np.min(data[:, i]) > -0.0000001:
            defpos_nr += 1
            defpos_indexes = defpos_indexes + [i]
    nr_data, nr_columns = data.shape
    nr_features = nr_columns**2 + order*nr_columns + 2*defpos_nr + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    for o in range(3, order + 1):
        for f in range(0, nr_columns):
            features[:, nr_columns ** 2 + (o-2)*nr_columns + f] = data[:, f] ** o

    # cubic root terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + (order-1) * nr_columns + f] = np.cbrt(data[:, f])

    # log terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + order * nr_columns + f] = np.log(data[:, defpos_indexes[f]] + 1)

    # square root terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + order * nr_columns + defpos_nr + f] = np.sqrt(data[:, defpos_indexes[f]])

    warnings.filterwarnings('error')
    # Whitening features
    for f in range(0, nr_features-1):
        try:
            features[:, f] = (features[:, f] - means[f]) / stds[f]
        except Warning:
            print(f)
            print(np.mean(features[:, f]))
            print(np.std(features[:, f]))

    # Add bias
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features


def build_poly(x, degree):
    """
    Polynomial basis functions for input data x, for j=0 up to j=degree.
    If there is a missing value -999 it does not change it.

        Args: 
        x : np.array
        Training dataset.
        
        degree : np.array
        Maximal degree of polynomial.

    Returns:
        a : np.array
        Dataset with polynomial attributes.
    """
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
    return (x[indicies[:index_number], ], x[indicies[index_number:], ], y[indicies[:index_number], ],
            y[indicies[index_number:], ])


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]
    
    return x, mean_x, std_x


def normalize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x   
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x > 0] = x[:, std_x > 0] / std_x[std_x > 0]
    
    x = x/(np.amax(x, axis=0) - np.amin(x, axis = 0))
    return x, mean_x, std_x       


def accuracy(weights, features, targets, nr_traindata, model_type):
    """
    Return accuracy value for given dataset and model type
    
    Args: 
        weights : np.array
        Weights of a model.
        
        features : np.array
        Dataset to compute the accuracy.

        targets : np.array
        Labels or values for the dataset.

        nr_traindata : int
        Number of samples in the dataset.

        model_type : string
        Model for which the accuracy is measured ["linear"|"logistic"|"logistic_cv"]

    Returns:
        np.array
        Accuracy of the model
    """
    if model_type == "linear":
        train_predictions = predict_labels(weights, features)
    elif model_type == "logistic":
        train_predictions = predict_labels_lg(weights, features)
    elif model_type == "logistic_cv":
        train_predictions = predict_labels_lg_cv(weights, features)
    return 1-(nr_traindata-train_predictions.dot(targets))/(2*nr_traindata)


def cross_validation(y, x, k_indices, k, lambda_, model_type, max_iters = 1000, gamma = 0.01):
    """
    Perform k-fold cross-validation and return the accuracy of ridge regression
    or logistic regression based on model_type variable.

    Args: 
        y : np.array
        Labels/targets for dataset.
        
        x : np.array
        Dataset without labels.

        k_indicies : np.array
        Array containing indicies of samples for each CV.

        k : int
        Number of folds of CV.

        model_type : string
        Model for which the accuracy is measured ["linear"|"logistic"|"logistic_cv"]

        max_iters : int
        Maximal number of iterations of algorithm.
        
        gamma : float
        Learning rate of the gradient descent.

    Returns:
        acc_tr, acc_te : (float, float)
        acc_tr : Accuracy of the model on train dataset
        acc_te : Accuracy of the model on test dataset
    """
    acc_tr_arr = np.zeros(k)
    acc_te_arr = np.zeros(k)
    for i in range(k):
        mask = np.zeros(y.shape[0], dtype=bool)
        mask[k_indices[i]] = True
        x_train = x[~mask]
        y_train = y[~mask]
        x_test = x[mask]
        y_test = y[mask]      
        
        initial_w = np.random.uniform(-0.3, 0.3, x_train.shape[1])
        
        if(model_type == "linear"):
            weights, _ = ridge_regression(y_train, x_train, lambda_)
        elif(model_type == "logistic_cv"):
            weights, _ = reg_logistic_regression(y_train, x_train, lambda_, initial_w, max_iters, gamma)
        acc_tr_arr[i] = accuracy(weights, x_train, y_train, x_train.shape[0], model_type)
        acc_te_arr[i] = accuracy(weights, x_test, y_test, x_test.shape[0], model_type)
        
    acc_tr = np.mean(acc_tr_arr)
    acc_te = np.mean(acc_te_arr)
    return acc_tr, acc_te


def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train accuracy')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test accuracy')
    plt.xlabel("lambda")
    plt.ylabel("accuracy")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    plt.show()


def plot_corr_matrix(corr_matrix, labels):
    """
    Plot correlation matrix for given dataset and its headers
    
    Args:
        corr_matrix : np.array
        Correlation matrix

        labels : list of strings
        Names of variables in correlation matrix
    """
    fig_cor, axes_cor = plt.subplots(1,1)
    fig_cor.set_size_inches(12, 12)

    myimage = axes_cor.imshow(corr_matrix, cmap='seismic', interpolation='nearest', vmax=1, vmin=-1)
    plt.colorbar(myimage)

    axes_cor.set_xticks(np.arange(0, corr_matrix.shape[0], corr_matrix.shape[0]*1.0/len(labels)))
    axes_cor.set_yticks(np.arange(0, corr_matrix.shape[1], corr_matrix.shape[1]*1.0/len(labels)))

    axes_cor.set_xticklabels(labels)
    axes_cor.set_yticklabels(labels)
    plt.xticks(rotation=90)
    plt.draw()


def replace_999_by_mean(data):
    """Insert attribute mean in place of (-999) values without counting them to mean"""
    for i in range(data.shape[1]):
        data[data[:, i] == -999, i] = np.nan
        data[np.isnan(data[:, i]), i] = np.nanmean(data[:, i])
        

def insert_median_for_nan(data):
    """Insert attribute median in place of (-999) values without counting them to median"""
    for i in range(data.shape[1]):
        data[data[:, i] == -999, i] = np.nan
        data[np.isnan(data[:, i]), i] = np.nanmedian(data[:, i])


def insert_function_for_missing(data, func):
    """Insert value given by the func in place of (-999) values"""
    for i in range(data.shape[1]):
        data[data[:, i] == -999, i] = np.nan
        data[np.isnan(data[:, i]), i] = func(data[:, i])
    

# Functions for linear regression task

def sgd_step(target, features, weights, gamma):
    """One step of weight update for stochastic gradient descent"""
    error = features.dot(weights)-target
    new_weights = weights - gamma * error * features
    return new_weights


def ridge_regression(y, tx, lambda_):
    """Regularized linear regression using normal equations"""
    xTx = np.dot(tx.T, tx)
    shape = np.shape(xTx)
    a = xTx + lambda_*(2.0*len(y)) * np.identity(shape[0])
    b = np.dot(tx.T, y)
    weights = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, weights)
    return weights, loss


def least_squares(y, tx):
    """Linear regression using normal equations

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

    Returns:
        (weights, loss) : (np.array, float) 
        
        weights : Final weights for the model.
        loss : Final loss value.     
    """
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    weights = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, weights)
    return weights, loss


def compute_mse(y, tx, w):
    """
    Compute the loss by mse.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

    Returns:
        mse : float
        Final loss value.     
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_loss(y, tx, w):
    """
    Compute the loss by mse.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

    Returns:
        mse : float
        Loss value. 
    """
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def compute_gradient(y, tx, w, lambda_ = 0):
    """
    Compute the gradient.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

        lambda_ : float
        Regularization parameter

    Returns:
        gradient : float
        Gradient of a loss function. 
    """
    error = y - tx.dot(w)
    return (-1/y.shape[0]) * np.dot(tx.T, error.T) + 2*lambda_*w


def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_ = 0):
    """
    Gradient descent algorithm.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        initial_w : np.array
        Initial weights for algorithm

        gamma : float
        Learning rate of the gradient descent.

        lambda_ : float
        Regularization parameter

    Returns:
        (losses, ws) : (np.array, np.array)
        losses - array of losses during optimization process
        ws - array of weights from every step of the algorithm.
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 0
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        w = w - gamma*compute_gradient(y, tx, w, lambda_)
        ws.append(w)
        losses.append(loss)
        if(n_iter%1000 == 0):
            print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        if(len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold):
            break
    return losses, ws


def compute_stoch_gradient(y, tx, w, lambda_):
    """
    Compute a stochastic gradient from just few examples n and their corresponding y_n labels.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

        lambda_ : float
        Regularization parameter

    Returns:
        gradient : float
        Gradient of a loss function. 
    """
    return compute_gradient(y, tx, w, lambda_)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, lambda_ = 0):
    """
    Stochastic gradient descent algorithm for linear regression.

        Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        initial_w : np.array
        Initial weights for algorithm.

        batch_size : int
        Batch size used to compute gradient of a function.

        max_iters : int
        Maximal number of iterations of algorithm.

        gamma : float
        Learning rate of the gradient descent.

        lambda_ : float
        Regularization parameter

    Returns:
        (losses, ws) : (np.array, np.array)
        losses - array of losses during optimization process
        ws - array of weights from every step of the algorithm.
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 0

    for n_iter in range(max_iters):              
        n = np.random.random_integers(size=batch_size, low=0, high=y.shape[0] - 1)
        w = w - gamma*compute_stoch_gradient(y[n], tx[n], w, lambda_)
        ws.append(w)
        loss = compute_loss(y, tx, w)
        losses.append(loss)
        if(n_iter%1000 == 0):
            print("SGD({bi}/{ti}): loss={l}, norm of weights={w}, gamma={g}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w = w.dot(w), g = gamma))
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses, ws

# Functions for logistic regression

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1 + np.exp(t))


def calculate_loss(y, tx, w):
    """
    Compute the cost by negative log likelihood.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

    Returns:
        loss : float
        Loss value. 
    """
    return np.sum(np.log(1 + np.exp(tx.dot(w))) - y * tx.dot(w))


def calculate_gradient(y, tx, w, lambda_ = 0):
    """
    Compute the gradient of loss for logistic regression.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

        lambda_ : float
        Regularization parameter

    Returns:
        gradient : float
        Gradient of a loss function.
    """
    return tx.T.dot((sigmoid(tx.dot(w)) - y)) + lambda_ * np.abs(w)


def learning_by_gradient_descent(y, tx, w, gamma, lambda_ = 0):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        gamma : float
        Learning rate of the gradient descent.

        lambda_ : float
        Regularization parameter

    Returns:
        (loss, w) : (float, np.array)
        loss - loss value
        w - weights after one step of GD.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w, lambda_)
    w = w - gamma*grad
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression algorithm

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        lambda_ : float
        Regularization parameter.

        initial_w : np.array
        Initial weights for algorithm.

        max_iters : int
        Maximal number of iterations of algorithm.

        gamma : float
        Learning rate of the gradient descent.

    Returns:
        (weights, loss) : (np.array, float)
        weights - final weights after optimization
        loss - value of loss.
    """
    weights = initial_w
    batch_size = 1
    debug = False
    for i in range(max_iters):
        #n = np.random.random_integers(size = batch_size, low = 0, high = y.shape[0] - 1)
        #_, weights = learning_by_gradient_descent(y[n], tx[n], weights, gamma, lambda_)
        #_, weights = learning_by_penalized_gradient(y, tx, weights, gamma, lambda_)
        loss, weights = learning_by_newton_method(y, tx, weights, gamma, lambda_)
        
        if(i%1000 == 0 and debug == True):
            loss = calculate_loss(y, tx, weights)
            print("Iter({bi}/{ti}): loss={l}, wieghts = {w}, gamma={g}".format(
              bi=i, ti=max_iters - 1, l=loss, g = gamma, w = weights))            
        
    loss = calculate_loss(y, tx, weights)
    return weights, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression algorithm.

        Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        initial_w : np.array
        Initial weights for algorithm.

        max_iters : int
        Maximal number of iterations of algorithm.

        gamma : float
        Learning rate of the gradient descent.

    Returns:
        (weights, loss) : (np.array, float)
        weights - final weights after optimization
        loss - value of loss.
        """
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)
         

def calculate_hessian(y, tx, w):
    """
    Return the hessian of the loss function.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : np.array
        Weights of a model

    Returns:
        hessian : np.array
        Hessian of the loss function. 
    """
    Sig = sigmoid(np.dot(tx, w))
    S = (Sig*(1-Sig)).flatten()
    return np.dot((tx.T)*S, tx)


def logistic_regression_newton(y, tx, w, lambda_ = 0):
    """
    Return the loss, gradient, and hessian.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : float
        Weights of a model.

        lambda_ : float
        Regularization parameter.

    Returns:
        (loss, gradient, hessian) : (float, np.array, np.array)
        loss - Value of loss.
        gradient - Gradient of a loss function.
        hessian - Hessian of the loss function.
    """
    loss = calculate_loss(y, tx, w) + lambda_* np.dot(w.T, w)
    gradient = calculate_gradient(y, tx, w, lambda_)
    hessian = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w, gamma, lambda_ = 0):
    """
    Do one step on Newton's method. Return the loss and updated w.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : float
        Weights of a model.

        gamma : float
        Learning rate of the algorithm.

        lambda_ : float
        Regularization parameter.

    Returns:
        (loss, w) : (np.array, float)
        weights - final weights after optimization
        loss - value of loss.
    """
    loss, gradient, hessian = logistic_regression_newton(y, tx, w, lambda_)
    w = w - gamma*np.linalg.inv(hessian).dot(gradient)
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """
    Return the loss, gradient, and hessian.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        gamma : float
        Learning rate of the gradient descent.

        lambda_ : float
        Regularization parameter

    Returns:
        (loss, gradient, hessian) : (float, np.array, np.array)
        loss - Value of loss.
        gradient - Gradient of a loss function.
        hessian - Hessian of the loss function.
    """
    loss, gradient, hessian = logistic_regression_newton(y, tx, w)
    reg = (lambda_/2) * w.T.dot(w)
    return loss+reg, gradient + 2*lambda_*np.abs(w), hessian + 2*lambda_*np.eye(w.shape[0])


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args: 
        y : np.array
        Labels/targets vector.
        
        tx : np.array
        Training dataset.

        w : float
        Weights of a model.

        gamma : float
        Learning rate of the algorithm.

        lambda_ : float
        Regularization parameter.

    Returns:
        (loss, w) : (np.array, float)
        weights - final weights after optimization
        loss - value of loss.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*np.linalg.inv(hessian).dot(gradient)
    return loss, w