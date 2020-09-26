import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,LeaveOneOut
from scipy.io import loadmat
import pandas as pd
mat = loadmat('Data_PTC_vs_FTC.mat')
mat = mat['Data'][0][0][0].T
train = pd.DataFrame({'c1':mat[:,1],'c2':mat[:,788]})
y = pd.DataFrame({'c3':mat[:,2]})
params = {
 "objective" : "regression",
 "metric" : {"mae"},
 "num_leaves" : 9,
 "learning_rate" : 0.09,
}
num_folds=3
kf = KFold(n_splits=num_folds, random_state=None, shuffle=False)
train=np.array(train)
y=np.array(y).ravel()
validError = 0
for train_index, test_index in kf.split(train):
 X_train, X_test = train[train_index], train[test_index]
 y_train, y_test = y[train_index], y[test_index]
 trn_data = lgb.Dataset(X_train, y_train)
 val_data = lgb.Dataset(X_test, y_test)
 evals_result = {}
 model = lgb.train( params,
 train_set = trn_data,
 num_boost_round=100000,
 early_stopping_rounds=100,
 verbose_eval=2000,
 valid_sets=[trn_data,val_data],
 evals_result=evals_result
 )
 val_pred = model.predict(X_test, num_iteration = model.best_iteration)
 validError += mean_absolute_error(y_test,val_pred)
 print('Plot metrics')
 ax = lgb.plot_metric(evals_result, metric='l1')
 plt.show()
print("Error: ", validError/num_folds)
loo = LeaveOneOut()
num_folds = loo.get_n_splits(train)
for train_index, test_index in loo.split(train):
 X_train, X_test = train[train_index], train[test_index]
 y_train, y_test = y[train_index], y[test_index]
 trn_data = lgb.Dataset(X_train, y_train)
 val_data = lgb.Dataset(X_test, y_test)
 evals_result = {}
 model = lgb.train( params,
 train_set = trn_data,
 num_boost_round=100000,
 early_stopping_rounds=100,
 verbose_eval=2000,
 valid_sets=[trn_data,val_data],
 evals_result=evals_result
 )
 val_pred = model.predict(X_test, num_iteration = model.best_iteration)
 validError += mean_absolute_error(y_test,val_pred)
 print('Plot metrics')
 ax = lgb.plot_metric(evals_result, metric='l1')
 plt.show()
print("Error: ", validError/num_folds)
