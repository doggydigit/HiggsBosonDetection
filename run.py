import numpy as np
import sys
sys.path.insert(0, 'scripts')
from proj1_helpers import *
from implementations import *

# Load Training Data
targets, data, ids = load_csv_data("data/train.csv", False)

# Preprocess Train Data
max_pol = 16
data = add_mass_binary(data)
replace_999_by_mean(data)
(data0, data1, data2, data3, labels0, labels1, labels2, labels3) = split_data_by_jet_num(data, False, targets)
(data0, means0, stds0) = build_features(data0, max_pol)
(data1, means1, stds1) = build_features(data1, max_pol)
(data2, means2, stds2) = build_features(data2, max_pol)
(data3, means3, stds3) = build_features(data3, max_pol)

# Train Models
lambda_ = 10**(-13)
weights0 = ridge_regression(labels0, data0, lambda_=lambda_)
weights1 = ridge_regression(labels1, data1, lambda_=lambda_)
weights2 = ridge_regression(labels2, data2, lambda_=lambda_)
weights3 = ridge_regression(labels3, data3, lambda_=lambda_)

# Delete some of the big training variables to free some memory in case your computer has less RAM than ours :)
del data0
del data1
del data2
del data3

# Load Test Data
test_targets, test_data, test_ids = load_csv_data("data/test.csv", False)

# Preprocess Test Data
test_data = add_mass_binary(test_data)
replace_999_by_mean(test_data)
(test_data0, test_data1, test_data2, test_data3, masks0, masks1, masks2, masks3) = split_data_by_jet_num(test_data)
test_data0 = build_features(test_data0, max_pol, True, means0, stds0)
test_data1 = build_features(test_data1, max_pol, True, means1, stds1)
test_data2 = build_features(test_data2, max_pol, True, means2, stds2)
test_data3 = build_features(test_data3, max_pol, True, means3, stds3)

# Use model to predict labels of test set
test_predictions0 = predict_labels(weights0[0], test_data0[0])
test_predictions1 = predict_labels(weights1[0], test_data1[0])
test_predictions2 = predict_labels(weights2[0], test_data2[0])
test_predictions3 = predict_labels(weights3[0], test_data3[0])

# Create Submission
prediction = np.ones(len(test_ids))
prediction[masks0] = test_predictions0
prediction[masks1] = test_predictions1
prediction[masks2] = test_predictions2
prediction[masks3] = test_predictions3
create_csv_submission(test_ids, prediction, "Kozak_Nurmi_Tsai_Submission")
