import argparse
import numpy as np
import matplotlib as mpl
import os

from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

mpl.use("Qt5Agg")

from os.path import dirname, join as pjoin
import scipy.io as sio

from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
    MultiRocket,
    MultiRocketMultivariate,
)

# This is the path where all the files are stored.
folder_path = '/Users/yao-chiyu/OneDrive - Washington University in St. Louis/NIPS2023/Public code/'
noise_level = ["Inf", "20", "10", "5", "2"];

# 2.3 Initialise MiniRocket and Transform the Training Data
minirocket_multi = MiniRocketMultivariate()
Y1 = 0.*np.ones((5,1))
Y2 = 1.*np.ones((5,1))
Y3 = 2.*np.ones((5,1))
Y4 = 3.*np.ones((5,1))
Y5 = 4.*np.ones((5,1))
Y = np.ravel(np.concatenate((Y1, Y2, Y3, Y4, Y5)))
y_train = Y
y_test = Y
# Open one of the files,
MiniRocket_ACC = []
RF_ACC = []
HC2_ACC = []
num_features = 10000
verbose = 2
for data_file in sorted(os.listdir(folder_path)):
    if data_file.endswith("N500T200.mat"):
        mat_fname = pjoin(folder_path, data_file)
        mat_contents = sio.loadmat(mat_fname, mat_dtype=True)
        X = mat_contents['X']
        arr_3d = X.transpose((2,0,1))
        arr_3d_train = arr_3d[0:25,:,0:-1]
        arr_3d_test = arr_3d[25:50,:, 0:-1]

        minirocket_multi.fit(arr_3d_train)
        X_train_transform = minirocket_multi.transform(arr_3d_train)
        # 2.4 Fit a Classifier
        scaler = StandardScaler(with_mean=False)
        X_train_scaled_transform = scaler.fit_transform(X_train_transform)

        classifier = RidgeClassifierCV(alphas=np.logspace(-5, 5, 20))
        classifier.fit(X_train_scaled_transform, y_train)

        # 2.5 Load and Transform the Test Data
        X_test_transform = minirocket_multi.transform(arr_3d_test)

        # 2.6 Classify the Test Data
        X_test_scaled_transform = scaler.transform(X_test_transform)
        accuracy = classifier.score(X_test_scaled_transform, y_test)
        MiniRocket_ACC.append(accuracy)

print(MiniRocket_ACC)