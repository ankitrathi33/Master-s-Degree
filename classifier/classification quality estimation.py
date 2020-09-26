import pandas as pd
import numpy as np
import mat4py
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
###sample dataset
mean_class_1 = [0, 0]
covariance_class_1 = [[2, -1], [-1, 2]] # diagonal covariance-class 1
class_1_label = [1 for i in range(0, 200)]
mean_class_2 = [2, 2]
covariance_class_2 = [[1, 0], [0, 1]] # diagonal covariance-class 2
class_2_label = [0 for i in range(0, 200)]
class_1 = np.random.multivariate_normal(mean_class_1, covariance_class_1, 200)
class_2 = np.random.multivariate_normal(mean_class_2, covariance_class_2, 200)
dataset_class_one = pd.DataFrame({'X1': class_1[:, 0], 'X2': class_1[:, 1],
'y': class_1_label})
dataset_class_two = pd.DataFrame({'X1': class_2[:, 0], 'X2': class_2[:, 1],
'y': class_2_label})
dataset = dataset_class_one.append(dataset_class_two)
#
#plt.scatter(dataset['X1'], dataset['X2'], c=dataset['y'], cmap='viridis_r')
#
#plt.show()
# plotting ROC CURVE
def plot_roc_curve(fpr, tpr):
plt.plot(fpr, tpr, color='orange', label='ROC') plt.plot([0, 1], [0,
1], color='darkblue', linestyle='--') plt.xlabel('False Positive
Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()
def cross_validation(estimator, number_of_folds: int, dataset_X: np.array,
dataset_y: np.array):
accuracies = []
dataset_X, dataset_y = shuffle(dataset_X, dataset_y)
kf = KFold(n_splits=number_of_folds)
kf.get_n_splits(dataset_X)
for train_index, test_index in kf.split(dataset_X):
# splitting data
X_train, X_test = dataset_X[train_index], dataset_X[test_index]
y_train, y_test = dataset_y[train_index], dataset_y[test_index]
# generating classifier and calculation of accuracy
estimator.fit(X_train, y_train.ravel())
prediction = estimator.predict(X_test)
accuracies.append(accuracy_score(y_test, prediction))
return accuracies
def leave_one_out(estimator, dataset_X: np.array, dataset_y: np.array):
accuracies = []
dataset_X, dataset_y = shuffle(dataset_X, dataset_y)
loo = LeaveOneOut()
loo.get_n_splits(dataset_X)
for train_index, test_index in loo.split(dataset_X):
# print("TRAIN:", train_index, "TEST:", test_index)
X_train, X_test = dataset_X[train_index], dataset_X[test_index]
y_train, y_test = dataset_y[train_index], dataset_y[test_index]
# generating classifier and calculation of accuracy
estimator.fit(X_train, y_train.ravel())
prediction = estimator.predict(X_test)
accuracies.append(accuracy_score(y_test, prediction))
return accuracies
def calulate_qualities(cm):
print()
print("CONFUSION MATRIX: ")
Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP +
FN)
FN)
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN +
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] +
cm[1][1]) # Accuracy= TP + TN / (TP + TN + FP + FN)
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] +
cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP)
print("Sensitivity: ", Sensitivity) print("Specificity:
", Specificity) print("Accuracy: ", Accuracy)
print("Error: ", Error)
######################### ARTFICIAL DATA
###########################################
# getting data
X = np.array(dataset[['X1', 'X2']])
y = np.array(dataset[['y']])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# creating classifier K-fold
classifier_K_fold = SVC(kernel='rbf', random_state=0, gamma='auto',
probability=True)
classifier_K_fold.fit(X_train, y_train)
# creating classifier Leave one out
classifier_Leave_one_out = SVC(kernel='rbf', random_state=0, gamma='auto',
probability=True)
classifier_Leave_one_out.fit(X_train, y_train)
#calculating accuracy_score
accuracy_score_K_fold = cross_validation(estimator = classifier_K_fold,
number_of_folds=10, dataset_X=X_train, dataset_y=y_train)
accuracy_score_Leave_One_Out =
leave_one_out(estimator=classifier_Leave_one_out, dataset_X=X_train,
dataset_y=y_train)
#prediction of estimators
prediction_K_fold = classifier_K_fold.predict(X_test)
prediction_Leave_one_out = classifier_Leave_one_out.predict(X_test)
prediction1 = classifier_K_fold.predict_proba(X_test)
prediction1 = prediction1[:, 1]
prediction2 = classifier_K_fold.predict_proba(X_test)
prediction2 = prediction2[:, 1]
#confusion matrix
cm_K_fold = confusion_matrix(y_test, prediction_K_fold)
cm_Leave_one_out = confusion_matrix(y_test, prediction_Leave_one_out)
#ploting ROC curves
fpr, tpr, threshold = roc_curve(y_test, prediction1)
plot_roc_curve(fpr, tpr)
fpr, tpr, threshold = roc_curve(y_test, prediction2)
plot_roc_curve(fpr, tpr)
calulate_qualities(cm_K_fold)
calulate_qualities(cm_Leave_one_out)
######################################## REAL DATA
#########################################
data = mat4py.loadmat('Data_PTC_vs_FTC.mat')
D = pd.DataFrame(data['Data']['D'])
X = pd.DataFrame.transpose(pd.DataFrame(data['Data']['X']))
G = pd.DataFrame.transpose(pd.DataFrame(data['Data']['gene_names']))
X_data_var = pd.DataFrame(data['Data']['X'])
X_data = np.array(pd.DataFrame.transpose(pd.DataFrame(X_data_var.iloc[1:3,
:].values)))
Y_data = np.array(pd.DataFrame(D.iloc[:, 0].values))
plt.scatter(X_data[:, 0:1], X_data[:, 1:], c=Y_data, cmap='viridis_r')
plt.show()
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_data, Y_data,
test_size=0.25, random_state=0)
sc = StandardScaler()
X_train2 = sc.fit_transform(X_train2)
X_test2 = sc.transform(X_test2)
##creating classifier K-fold
classifier_Leave_one_out_R = SVC(kernel='rbf', random_state=0, gamma='auto',
probability=True)
classifier_Leave_one_out_R.fit(X_train2, y_train2)
# creating classifier Leave one out
classifier_K_fold_R = SVC(kernel='rbf', random_state=0, gamma='auto',
probability=True)
classifier_K_fold_R.fit(X_train2, y_train2)
accuracy_score_Leave_One_OutR =
leave_one_out(estimator=classifier_Leave_one_out_R, dataset_X=X_train2,
dataset_y=y_train2)
accuracy_score_K_foldR = cross_validation(estimator = classifier_K_fold_R,
number_of_folds=10, dataset_X=X_train2, dataset_y=y_train2)
prediction_K_fold_R = classifier_K_fold_R.predict(X_test2)
prediction_Leave_one_out_R = classifier_Leave_one_out_R.predict(X_test2)
prediction3 = classifier_Leave_one_out_R.predict_proba(X_test2)
prediction3 = prediction3[:, 1]
prediction4 = classifier_K_fold_R.predict_proba(X_test2)
prediction4 = prediction4[:, 1]
cm_K_fold_R = confusion_matrix(y_test2, prediction_K_fold_R)
cm_Leave_one_out_R = confusion_matrix(y_test2, prediction_Leave_one_out_R)
fpr, tpr, threshold = roc_curve(y_test2, prediction3)
plot_roc_curve(fpr, tpr)
fpr, tpr, threshold = roc_curve(y_test2, prediction4)
plot_roc_curve(fpr, tpr)
# calculation of qualities
calulate_qualities(cm_K_fold_R)
calulate_qualities(cm_Leave_one_out_R)
###################################################
###########################
#####
ax= plt.subplots(figsize=(5,5))
sns.set(font_scale=1.5)
sns_plot = sns.heatmap(cm_K_fold, cbar=False, square=True, annot=True,fmt='g')
ax= plt.subplots(figsize=(5,5))
sns.set(font_scale=1.5)
sns_plot = sns.heatmap(cm_Leave_one_out, cbar=False, square=True,
annot=True,fmt='g')
ax= plt.subplots(figsize=(5,5))
sns.set(font_scale=1.5)
sns_plot = sns.heatmap(cm_K_fold_R, cbar=False, square=True,
annot=True,fmt='g')
ax= plt.subplots(figsize=(5,5))
sns.set(font_scale=1.5)
sns_plot = sns.heatmap(cm_Leave_one_out_R, cbar=False, square=True, annot=True,fmt='g')

