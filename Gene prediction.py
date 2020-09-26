import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir())
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold,LeaveOneOut
from sklearn.metrics import roc_curve, auc

from  scipy.io import loadmat
import pandas as pd
mat = loadmat('Data_PTC_vs_FTC.mat')
mat = mat['Data'][0][0][0].T

train = pd.DataFrame({'c1':mat[:,1],'c2':mat[:,788]})
train['target'] = np.random.randint(1, 10, train.shape[0]) + pd.DataFrame({'c3':mat[:,2]})['c3']
train.loc[train.target > 10,'target'] = 0
train.loc[train.target != 0,'target'] = 1
print(train.target.value_counts())
y = train.target
train = train.drop('target',axis=1)

params = {
        "objective" : "regression", 
        "metric" : {"auc"}, 
        "num_leaves" : 9, 
        "learning_rate" : 0.00009, 
}

num_folds=3
kf = KFold(n_splits=num_folds, random_state=None, shuffle=False)
train=np.array(train)
y=np.array(y).ravel()

mean = sum(y/len(y))
y = np.where(y < mean, 0, 1)
print(y)
validError = 0
for train_index, test_index in kf.split(train):
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = y[train_index], y[test_index]

    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    evals_result = {} 
    model = lgb.train(  params, 
                        train_set = trn_data,
                        num_boost_round=100000,
                        early_stopping_rounds=100,
                        verbose_eval=2000, 
                        valid_sets=[trn_data,val_data],
                        evals_result=evals_result
                      )
    val_pred = model.predict(X_test, num_iteration = model.best_iteration)
    validError += mean_absolute_error(y_test,val_pred)
    fpr, tpr, thresholds = roc_curve(y_test, val_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy',  linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()
	
print("Error: ", validError/num_folds)
from sklearn.metrics import confusion_matrix
val_pred = np.where(val_pred < 0.5, 0, 1)
confusion_matrix(y_test, val_pred)

print("Error: ", validError/num_folds)
from sklearn.metrics import confusion_matrix
val_pred = np.where(val_pred < 0.5, 1, 0)
confusion_matrix(y_test, val_pred)

loo = LeaveOneOut()
num_folds = loo.get_n_splits(train)
for train_index, test_index in loo.split(train):
    X_train, X_test = train[train_index], train[test_index]
    y_train, y_test = y[train_index], y[test_index]

    trn_data = lgb.Dataset(X_train, y_train)
    val_data = lgb.Dataset(X_test, y_test)
    evals_result = {} 
    model = lgb.train(  params, 
                        train_set = trn_data,
                        num_boost_round=100000,
                        early_stopping_rounds=100,
                        verbose_eval=2000, 
                        valid_sets=[trn_data,val_data],
                        evals_result=evals_result
                      )
    val_pred = model.predict(X_test, num_iteration = model.best_iteration)
    validError += mean_absolute_error(y_test,val_pred)
	
print("Error: ", validError/num_folds)