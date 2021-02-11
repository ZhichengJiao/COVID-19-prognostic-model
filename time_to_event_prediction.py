import numpy as np
from sksurv.ensemble import RandomSurvivalForest
import hdf5storage as hds

# load the data set
data_all = hds.loadmat('path/to/data')  # The mat file contains clinical or image features and critical label

# obtain the feature and label variables from the data set
feat_train = data_all['feat_train']
feat_test = data_all['feat_test']
label_train = data_all['label_train']
label_test = data_all['label_test']

# define the labels in the format of RSF model
labels_crit_train = np.ndarray(shape=(label_train.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])
labels_crit_test = np.ndarray(shape=(label_test.shape[0], ), dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

# obtain the labels in the format of RSF model
# the fist column of labels are time to critical and the 2nd column is the label of critical or not

for i in range(labels_crit_train.shape[0]):
    if label_train[i, 1] == 1:
        labels_crit_train[i] = (True,  label_train[i, 0])
    else:
        labels_crit_train[i] = (False, label_train[i, 0])

for i in range(labels_crit_test.shape[0]):
    if label_test[i, 1] == 1:
        labels_crit_test[i] = (True,  label_test[i, 0])
    else:
        labels_crit_test[i] = (False, label_test[i, 0])


# Define the parameter of random survival forest
n_estimators = 'number of estimators'
max_depth = 'the max depth'
min_samples_split = 'num of min_samples_split'
min_samples_leaf = 'num of min_samples_leaf'
max_features = 'num of features to use'
bootstrap = True
rsf = RandomSurvivalForest(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap,
                           min_samples_split=min_samples_split,
                           min_samples_leaf=min_samples_leaf)

# fit the model
rsf.fit(feat_train, labels_crit_train)

# obtain the c-index on test data
result_test = rsf.score(feat_test, labels_crit_test)

# or if you would like to know the predicted risk scores
risks_test = rsf.predict(feat_test)
