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
    weights, losses = gradient_descent(y, tx, initial_w, max_iters, gamma, 0)
    return weights[-1], losses[-1]


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    weights, losses = stochastic_gradient_descent(y, tx, initial_w, 1, max_iters, gamma, 0)
    return weights[-1], losses[-1]


def remove_outliers(data, label, outlier_constant):
    for column in range(0, len(data[0])):
        a = np.array(data[:, column])
        upper_quartile = np.percentile(a, 95)
        lower_quartile = np.percentile(a, 5)
        iqr = (upper_quartile - lower_quartile) * outlier_constant
        quartileset = (lower_quartile - iqr, upper_quartile + iqr)
        mask = np.ndarray.tolist(np.where(data[:, column] < quartileset[0])[0]) + np.ndarray.tolist(np.where(data[:, column] > quartileset[1])[0])
        data = np.delete(data, mask, axis=0)
        label = np.delete(label, mask, axis=0)
    return data, label


def add_mass_binaries(data):
    nrdata, nrcolumns = data.shape
    newdata = np.zeros((nrdata, nrcolumns+1))
    mask = data[:, 0] == -999
    newdata[:, 0:nrcolumns] = data
    newdata[:, -1] = mask.astype(int)
    return newdata


def split_data_by_jet_num(data, labels):
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
    labels0 = labels[m0]
    splitdata0 = splitdata0[:, np.ndarray.tolist(np.where(jetmask0)[0])]
    splitdata1 = data[m1, :]
    labels1 = labels[m1]
    splitdata1 = splitdata1[:, np.ndarray.tolist(np.where(jetmask1)[0])]
    splitdata2 = data[m2, :]
    labels2 = labels[m2]
    splitdata2 = splitdata2[:, np.ndarray.tolist(np.where(jetmask2)[0])]
    splitdata3 = data[m3, :]
    labels3 = labels[m3]
    splitdata3 = splitdata3[:, np.ndarray.tolist(np.where(jetmask3)[0])]

    return splitdata0, splitdata1, splitdata2, splitdata3, labels0, labels1, labels2, labels3


def add_sin_cos(data, nr_columns, nr_data):
    radian_indexes = [15, 18, 20, 25, 28]
    newdata = np.zeros((nr_data, nr_columns + 2*len(radian_indexes)))
    newdata[:, :nr_columns] = data
    n = 0
    for f in radian_indexes:
        newdata[:, nr_columns+n] = np.sin(data[:, f])
        n += 1
        newdata[:, nr_columns + n] = np.cos(data[:, f])
        n += 1
    return newdata


def second_order_features(data):
    nr_data, nr_columns = data.shape
    warnings.filterwarnings('error')
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
        try:
            features[:, f] = (features[:, f] - np.mean(features[:, f])) / np.std(features[:, f])
        except Warning:
            print(f, np.std(features[:, f]))

    # Add bias
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features


def second_order_features_and_4(data):
    defpos_nr = 0
    defpos_indexes = []
    for i in range(0, len(data[0])):
        if np.min(data[:, i]) > -0.0000001:
            defpos_nr += 1
            defpos_indexes = defpos_indexes + [i]
    nr_data, nr_columns = data.shape
    nr_features = nr_columns**2 + 8*nr_columns + 3*defpos_nr + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    # cubic order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + nr_columns + f] = data[:, f]**3

    # forth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 2*nr_columns + f] = data[:, f]**4

    # fifth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 3 * nr_columns + f] = data[:, f]**5

    # sixth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 4 * nr_columns + f] = data[:, f]**6

    # seventh order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 5 * nr_columns + f] = data[:, f] ** 7

    # eighth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 6 * nr_columns + f] = data[:, f] ** 8

    # ninth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 7 * nr_columns + f] = data[:, f] ** 9


    # log terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + 8 * nr_columns + f] = np.log(data[:, defpos_indexes[f]] + 1)

    # square root terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + 8 * nr_columns + defpos_nr + f] = np.sqrt(data[:, defpos_indexes[f]])

    # cubic root terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + 8 * nr_columns + 2*defpos_nr + f] = np.cbrt(data[:, defpos_indexes[f]])

    warnings.filterwarnings('error')
    # Whitening features
    means = np.zeros(nr_features)
    stds = np.zeros(nr_features)
    for f in range(0, nr_features-1):
        means[f] = np.mean(features[:, f])
        stds[f] = np.std(features[:, f])
        try:
            features[:, f] = (features[:, f] - means[f]) / stds[f]
        except Warning:
            print(f)
            print(np.mean(features[:, f]))
            print(np.std(features[:, f]))

    # Add bias
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features, means, stds


def second_order_features_and_4_for_test(data, means, stds):
    defpos_nr = 0
    defpos_indexes = []
    for i in range(0, len(data[0])):
        if np.min(data[:, i]) > -0.0000001:
            defpos_nr += 1
            defpos_indexes = defpos_indexes + [i]
    nr_data, nr_columns = data.shape
    nr_features = nr_columns**2 + 8*nr_columns + 3*defpos_nr + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    # cubic order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + nr_columns + f] = data[:, f]**3

    # forth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 2*nr_columns + f] = data[:, f]**4

    # fifth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 3 * nr_columns + f] = data[:, f]**5

    # sixth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 4 * nr_columns + f] = data[:, f]**6

    # seventh order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 5 * nr_columns + f] = data[:, f] ** 7

    # eighth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 6 * nr_columns + f] = data[:, f] ** 8

    # ninth order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + 7 * nr_columns + f] = data[:, f] ** 9


    # log terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + 8 * nr_columns + f] = np.log(data[:, defpos_indexes[f]] + 1)

    # square root terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + 8 * nr_columns + defpos_nr + f] = np.sqrt(data[:, defpos_indexes[f]])

    # cubic root terms
    for f in range(0, defpos_nr):
        features[:, nr_columns ** 2 + 8 * nr_columns + 2*defpos_nr + f] = np.cbrt(data[:, defpos_indexes[f]])

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


def second_order_features_and_cubic(data):
    nr_data, nr_columns = data.shape
    nr_features = nr_columns**2 + 2*nr_columns + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    # cubic order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + nr_columns + f] = data[:, f]**3

    warnings.filterwarnings('error')
    # Whitening features
    means = np.zeros(nr_features)
    stds = np.zeros(nr_features)
    for f in range(0, nr_features-1):
        means[f] = np.mean(features[:, f])
        stds[f] = np.std(features[:, f])
        try:
            features[:, f] = (features[:, f] - means[f]) / stds[f]
        except Warning:
            print(f)
            print(np.mean(features[:, f]))
            print(np.std(features[:, f]))

    # Add bias
    features[:, nr_features-1] = np.ones([nr_data, 1])[:, 0]
    return features, means, stds


def second_order_features_and_cubic_for_test(data, means, stds):
    nr_data, nr_columns = data.shape
    nr_features = nr_columns**2 + 2*nr_columns + 1
    features = np.zeros([nr_data, nr_features])

    # second order terms
    for f1 in range(0, nr_columns):
        for f2 in range(0, nr_columns):
            features[:, f1*nr_columns + f2] = np.multiply(data[:, f1], data[:, f2])

    # first order terms
    for f in range(0, nr_columns):
        features[:, nr_columns**2 + f] = data[:, f]

    # cubic order terms
    for f in range(0, nr_columns):
        features[:, nr_columns ** 2 + nr_columns + f] = data[:, f]**3

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


# Additional funtions to manipulate the data
def white_cubic_features(data, nr_columns, nr_data):
    nr_features = 3*nr_columns + 1
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


def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    return x, mean_x, std_x


def normalize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x   
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    x = x/(np.amax(x, axis = 0) - np.amin(x, axis = 0))  
    return x, mean_x, std_x       


def accuracy(weights, features, targets, nr_traindata, model_type):
    """Return accuracy value for given dataset and model type"""
    if model_type == "linear":
        train_predictions = predict_labels(weights, features)
    elif model_type == "logistic":
        train_predictions = predict_labels_lg(weights, features)
    elif model_type == "logistic_cv":
        train_predictions = predict_labels_lg_cv(weights, features)
    return 1-(nr_traindata-train_predictions.dot(targets))/(2*nr_traindata)


def cross_validation(y, x, k_indices, k, lambda_, model_type, max_iters = 1000, gamma = 0.01):
    """return the accuracy of ridge regression or logistic regression based on model_type variable."""
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
        #loss_tr_arr[i] = np.sqrt(2 * compute_mse(y_train, x_train, weights))
        acc_tr_arr[i] = accuracy(weights, x_train, y_train, x_train.shape[0], model_type)
       # loss_te_arr[i] = np.sqrt(2 * compute_mse(y_test, x_test, weights))
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
    """Plot correlation matrix for given dataset and its headers"""
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
    loss = compute_mse(y,tx,weights)
    return weights, loss


def least_squares(y, tx):
    """Linear regression using normal equations"""
    a = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    weights = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, weights)
    return weights, loss


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
    threshold = 1e-8
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        w = w - gamma*compute_gradient(y, tx, w, lambda_)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
        if(np.abs(losses[-1] - losses[-2]) < threshold):
            break
    return losses, ws


def compute_stoch_gradient(y, tx, w, lambda_):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return compute_gradient(y, tx, w, lambda_)


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma, lambda_ = 0):
    """Stochastic gradient descent algorithm for linear regression."""
    ws = [initial_w]
    losses = []
    w = initial_w
    threshold = 1e-8

    for n_iter in range(max_iters):              
        n = np.random.random_integers(size = batch_size, low = 0, high = y.shape[0] - 1)
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

#def sigmoid(x):
#    "Numerically-stable sigmoid function."
#    return np.exp(-np.logaddexp(0, -x))


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
    grad = calculate_gradient(y, tx, w, lambda_)
    w = w - gamma*grad
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression algorithm"""
    weights = initial_w
    batch_size = 1
    prev_loss, prev_weights = -10, None
    debug = True
    threshold = 1e-8
    for i in range(max_iters):
        #n = np.random.random_integers(size = batch_size, low = 0, high = y.shape[0] - 1)
        #_, weights = learning_by_gradient_descent(y[n], tx[n], weights, gamma, lambda_)
        #_, weights = learning_by_penalized_gradient(y, tx, weights, gamma, lambda_)
        loss, weights = learning_by_newton_method(y, tx, weights, gamma, lambda_)
        
        if np.abs(loss - prev_loss) < threshold:
            weights = prev_weights
            break
            
        if(i%100 == 0 and debug == True):
            loss = calculate_loss(y, tx, weights)
            print("Iter({bi}/{ti}): loss={l}, wieghts = {w}, gamma={g}".format(
              bi=i, ti=max_iters - 1, l=loss, g = gamma, w = weights))
            
        prev_loss = loss    
        prev_weights = weights
        
    loss = calculate_loss(y, tx, weights)
    return weights, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression algorithm"""
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)
         

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    Sig_txw = sigmoid(np.dot(tx,w))
    S = (Sig_txw*(1-Sig_txw)).flatten()
    return np.dot((tx.T)*S,tx)


def logistic_regression_newton(y, tx, w, lambda_ = 0):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y, tx, w) + lambda_* np.dot(w.T, w)
    gradient = calculate_gradient(y, tx, w, lambda_)
    hessian = calculate_hessian(y, tx, w) + 2*lambda_*np.eye(w.shape[0])
    return loss, gradient, hessian


def learning_by_newton_method(y, tx, w, gamma, lambda_ = 0):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, gradient, hessian = logistic_regression_newton(y, tx, w, lambda_)
    w = w - gamma*np.linalg.inv(hessian).dot(gradient)
    return loss, w


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss, gradient, hessian = logistic_regression_newton(y, tx, w)
    reg = (lambda_/2) * w.T.dot(w)
    return loss+reg, gradient + 2*lambda_*np.abs(w), hessian + 2*lambda_*np.eye(w.shape[0])


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, gradient, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma*np.linalg.inv(hessian).dot(gradient)
    return loss, w