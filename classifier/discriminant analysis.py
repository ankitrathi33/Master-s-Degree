import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import scipy.stats from functools 
import reduce import math 
from sklearn.metrics import confusion_matrix 
import scipy.io 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn import metrics 
import scikitplot as skplt 
import pandas as pd 
import seaborn as sns 
from sklearn.datasets import make_classification 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.discriminant_analysis 
import LinearDiscriminantAnalysis as LDA 

data1 = pd.read_csv('dataset_1.csv',index_col=0) 
data2 = pd.read_csv('dataset_2.csv',index_col=0) 
plt.scatter(data1['x1'], data1['x2'], c=data1['y']) 
plt.show() 
plt.scatter(data2['x1'], data2['x2'], c=data2['y']) 
plt.show() 
#dataset 1 
sns_plot = sns.boxplot(x="y", y="x1", hue="y",data=data1, palette="coolwarm") 
sns_plot = sns.boxplot(x="y", y="x2", hue="y",data=data1, palette="coolwarm") 
#dataset 2 
sns_plot = sns.boxplot(x="y", y="x1", hue="y",data=data2, palette="coolwarm") 
sns_plot = sns.boxplot(x="y", y="x2", hue="y",data=data2, palette="coolwarm") 

#dataset 1 
g = sns.distplot(data1['x1'][data1['y'] == 0], kde=True,bins=60,kde_kws={"label": "distribution y = 0"}, hist_kws={"label": "histogram y = 0" }) 
sns.distplot(data1['x1'][data1['y'] == 1], kde=True,bins=60,kde_kws={"label": "distribution y = 1"}, hist_kws={"label": "histogram y = 1" }) 

g = sns.distplot(data1['x2'][data1['y'] == 0], kde=True,bins=60,kde_kws={"label": "distribution y = 0"}, hist_kws={"label": "histogram y = 0" }) 
sns.distplot(data1['x2'][data1['y'] == 1], kde=True,bins=60,kde_kws={"label": "distribution y = 1"}, hist_kws={"label": "histogram y = 1" }) 

#dataset 2 
g = sns.distplot(data2['x1'][data2['y'] == 0], kde=True,bins=60,kde_kws={"label": "distribution y = 0"}, hist_kws={"label": "histogram y = 0" }) 
sns.distplot(data2['x1'][data2['y'] == 1], kde=True,bins=60,kde_kws={"label": "distribution y = 1"}, hist_kws={"label": "histogram y = 1" }) 

g = sns.distplot(data2['x2'][data2['y'] == 0], kde=True,bins=60,kde_kws={"label": "distribution y = 0"}, hist_kws={"label": "histogram y = 0" }) 
sns.distplot(data2['x2'][data2['y'] == 1], kde=True,bins=60,kde_kws={"label": "distribution y = 0"}, hist_kws={"label": "histogram y = 0" }) 


X_train_linear, X_test_linear, y_train_linear, y_test_linear = train_test_split( data1[['x1','x2']], data1['y'], test_size=0.3, random_state=42) 

X_train_nonlinear, X_test_nonlinear, y_train_nonlinear, y_test_nonlinear = train_test_split( data2[['x1','x2']], data2['y'], test_size=0.3, random_state=42) 

def get_acc(true_val,predicted):
    matrix=confusion_matrix(true_val, predicted).ravel() 
    tn, fp, fn, tp = matrix 
    acc=(tn+tp)/(tp+tn+fp+fn) 
    return (acc,matrix,tp,fp,fn,tp) 

model_lda = LinearDiscriminantAnalysis() 
model_qda=QuadraticDiscriminantAnalysis() 
model_svm=SVC() 
model_lda.fit(X_train_linear, y_train_linear)
model_qda.fit(X_train_linear, y_train_linear) 
model_svm.fit(X_train_linear, y_train_linear) 
prediction_linear_data_lda=model_lda.predict(X_test_linear) 
prediction_linear_data_qda=model_qda.predict(X_test_linear) 
prediction_linear_data_svm=model_svm.predict(X_test_linear) 

def plot_roc_curve(fpr, tpr): 
    plt.plot(fpr, tpr, color='orange', label='ROC') 
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--') 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate') 
    plt.title('Receiver Operating Characteristic (ROC) Curve') 
    plt.legend() 
    plt.show() 
    
fpr, tpr, thresholds = roc_curve(y_test_linear, prediction_linear_data_lda) 
plot_roc_curve(fpr, tpr) 

fpr, tpr, thresholds = roc_curve(y_test_linear, prediction_linear_data_qda) 
plot_roc_curve(fpr, tpr) 

fpr, tpr, thresholds = roc_curve(y_test_linear, prediction_linear_data_svm) 
plot_roc_curve(fpr, tpr) 

cm = confusion_matrix(y_test_linear, prediction_linear_data_lda)
ax= plt.subplots(figsize=(10,10)) 
sns.set(font_scale=4) 
sns_plot = sns.heatmap(cm, cbar=False, square=True, annot=True,fmt='g') 

Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) 
print(Specificity) 
print(Accuracy) 
print(Error) 

auc = roc_auc_score(y_test_linear, prediction_linear_data_lda) 
print('AUC: %.4f' % auc) 

cm = confusion_matrix(y_test_linear, prediction_linear_data_qda) 
ax= plt.subplots(figsize=(10,10)) 
sns.set(font_scale=4) 
sns_plot = sns.heatmap(cm, cbar=False, square=True, annot=True,fmt='g') 
Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) 
print(Specificity) 
print(Accuracy) 
print(Error) 

auc = roc_auc_score(y_test_linear, prediction_linear_data_qda) 
print('AUC: %.4f' % auc) 

cm = confusion_matrix(y_test_linear, prediction_linear_data_svm) 
ax= plt.subplots(figsize=(10,10)) 
sns.set(font_scale=1.7) 
sns_plot = sns.heatmap(cm, cbar=False, square=True, annot=True,fmt='g') 
Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

auc = roc_auc_score(y_test_linear, prediction_linear_data_svm) 
print('AUC: %.4f' % auc) 

model_lda = LinearDiscriminantAnalysis() 
model_qda=QuadraticDiscriminantAnalysis() 
model_svm=SVC() 
model_lda.fit(X_train_nonlinear, y_train_nonlinear) 
model_qda.fit(X_train_nonlinear, y_train_nonlinear) 
model_svm.fit(X_train_nonlinear, y_train_nonlinear) 
prediction_nonlinear_data_lda=model_lda.predict(X_test_nonlinear) 
prediction_nonlinear_data_qda=model_qda.predict(X_test_nonlinear) 
prediction_nonlinear_data_svm=model_svm.predict(X_test_nonlinear) 
fpr, tpr, thresholds = roc_curve(y_test_nonlinear, prediction_nonlinear_data_lda) 
plot_roc_curve(fpr, tpr) 


Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

auc = roc_auc_score(y_test_nonlinear, prediction_nonlinear_data_lda) 
print('AUC: %.4f' % auc) 

fpr, tpr, thresholds = roc_curve(y_test_nonlinear, prediction_nonlinear_data_qda) 
plot_roc_curve(fpr, tpr) 

Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

auc = roc_auc_score(y_test_nonlinear, prediction_nonlinear_data_qda) 
print('AUC: %.4f' % auc) 

fpr, tpr, thresholds = roc_curve(y_test_nonlinear, prediction_nonlinear_data_qda) 
plot_roc_curve(fpr, tpr) 

Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

auc = roc_auc_score(y_test_nonlinear, prediction_nonlinear_data_qda) 
print('AUC: %.4f' % auc) 

fpr, tpr, thresholds = roc_curve(y_test_nonlinear, prediction_nonlinear_data_svm) 
plot_roc_curve(fpr, tpr) 

Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

auc = roc_auc_score(y_test_nonlinear, prediction_nonlinear_data_svm) 
print('AUC: %.4f' % auc) 

Titanic code: 
# Importing packages 
import pandas as pd 
import numpy as np 
import random as rnd 
# Importing visualization packages import seaborn as sns import matplotlib.pyplot as plt 
%matplotlib inline 
# Importing machine learning packages from sklearn.linear_model import LogisticRegression from sklearn.svm import SVC, LinearSVC from sklearn.ensemble import RandomForestClassifier from sklearn.neighbors import KNeighborsClassifier from sklearn.tree import DecisionTreeClassifier from sklearn.metrics import roc_curve from sklearn.metrics import roc_auc_score 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
df_train = pd.read_csv('train.csv') 
df_survived = pd.read_csv('gender_submission.csv') 
df_test = pd.read_csv('test.csv') 
combine = [df_train, df_test] 
pclass_age_grid = sns.FacetGrid(df_train, col='Survived',row='Pclass', size=2.2, aspect=1.6) 
pclass_age_grid.map(sns.distplot, "Age", hist=True, color="#0000DD") 
pclass_age_grid.add_legend() 
pclass_age_grid.set_ylabels('Number') 
df_train = df_train.drop(['Ticket', 'Cabin'], axis=1) 
df_test = df_test.drop(['Ticket', 'Cabin'], axis=1) 
combine = [df_train, df_test] 


for dataset in combine: 
    dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False) 
    pd.crosstab(df_train['Title'], df_train['Sex']) 
for dataset in combine: 
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss') 
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs') 
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Don', 'Sir', 'Jonkheer', 'Dona'],'Royalty') 
    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col','Dr','Major','Rev'],'Special') 

df_train[['Title','Survived']].groupby(['Title'], as_index=False).mean() 
title_mapping = {"Master": 1, "Miss": 2, "Mrs": 3, "Mr": 4, "Royalty": 5, "Special": 6} 
for dataset in combine: 
    dataset['Title'] = dataset['Title'].map(title_mapping) 
    dataset['Title'] = dataset['Title'].fillna(0) 
    
df_train = df_train.drop(['Name', 'PassengerId'], axis=1) 
df_test = df_test.drop(['Name'], axis=1) 
combine = [df_train, df_test] 
for dataset in combine: 
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int) 
    
df_train.head() 
pclass_sex_age_grid = sns.FacetGrid(df_train, row='Pclass', col='Sex') 
pclass_sex_age_grid.map(sns.distplot, "Age", hist=True, color="#0000DD") 
pclass_sex_age_grid.add_legend() 

df_train.Age.isnull().sum() 

median_age = np.zeros((2,3)) 

for dataset in combine: 
    for sex in range(0,2): 
        for pclass in range(0,3): 
            guess_df = dataset[(dataset['Sex'] == sex) & \ (dataset['Pclass'] == pclass+1)]['Age'].dropna() 
            age_guess = guess_df.median() median_age[sex,pclass] = age_guess median_age for dataset in combine: for i in range(0, 2): for j in range(0, 3): dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\ 'Age'] = median_age[i,j] 

dataset['Age'] = dataset['Age'].astype(int) 
df_train.head() 
df_train['AgeBand'] = pd.cut(df_train['Age'], 5) 
df_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True) 
for dataset in combine: 
    dataset.loc[ dataset['Age'] <= 16, 'AgeG'] = 0 
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'AgeG'] = 1 
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'AgeG'] = 2 
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'AgeG'] = 3 
    dataset.loc[ dataset['Age'] > 64, 'AgeG'] = 4 
    df_train.head() 
    
df_train = df_train.drop(['AgeBand','Age'], axis=1) 
combine = [df_train, df_test] 
df_train.head() 

for dataset in combine: 
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1 
    df_train[['FamilySize','Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False) 

df_train = df_train.drop(['SibSp', 'Parch'], axis=1) 
df_test = df_test.drop(['SibSp','Parch'], axis=1) 
combine = [df_train, df_test] df_train.head() 

freq_port = df_train.Embarked.dropna().mode()[0] 
freq_port 

for dataset in combine: 
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port) 
    
df_train[['Embarked','Survived']].groupby(['Embarked'], as_index = False).mean().sort_values(by='Survived', ascending=False) for dataset in combine: dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int) df_train.head() df_train = df_train.drop(['Fare'], axis=1) df_test = df_test.drop(['Fare'], axis=1) combine = [df_train, df_test] df_train.head() for dataset in combine: dataset.AgeG = dataset.AgeG.astype(int) df_train.head() 
df_test = df_test.drop(['Age'], axis=1) df_test.head() 
X_train = df_train.drop("Survived", axis=1) 
Y_train = df_train['Survived'] 
X_test = df_test.drop('PassengerId', axis=1).copy() 
svc = SVC() 
svc.fit(X_train, Y_train) 
Y_pred = svc.predict(X_test) 
acc_svm = round(svc.score(X_train, Y_train) *100 ,2) 
def plot_roc_curve(fpr, tpr): 
    plt.plot(fpr, tpr, color='orange', label='ROC') 
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--') 
    plt.xlabel('False Positive Rate') plt.ylabel('True Positive Rate') 
    plt.title('Receiver Operating Characteristic (ROC) Curve') plt.legend() 
    plt.show() 
    fig = plt.figure() 
    
fpr, tpr, thresholds = roc_curve(df_survived['Survived'], Y_pred) 
auc = roc_auc_score(df_survived['Survived'], Y_pred) 
print('AUC: %.5f' % auc) 

from sklearn.metrics import classification_report,confusion_matrix 
cm = confusion_matrix(df_survived['Survived'], Y_pred) 
print(confusion_matrix(df_survived['Survived'], Y_pred)) 

Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

qda_model=QDA() 
qda_model.fit(X_train, Y_train) 
Y_pred = qda_model.predict(X_test) 
fpr, tpr, thresholds = roc_curve(df_survived['Survived'], Y_pred) 
auc = roc_auc_score(df_survived['Survived'], Y_pred) 
print('AUC: %.5f' % auc) cm = confusion_matrix(df_survived['Survived'], Y_pred) 
print(confusion_matrix(df_survived['Survived'], Y_pred)) 
Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error) 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
lda=LDA() 
lda.fit(X_train, Y_train) 
Y_pred = lda.predict(X_test) 
auc = roc_auc_score(y_test, y_pred) 
print('AUC: %.5f' % auc) 
cm = confusion_matrix(df_survived['Survived'], Y_pred) 
print(confusion_matrix(df_survived['Survived'], Y_pred)) 
fpr, tpr, thresholds = roc_curve(df_survived['Survived'], Y_pred) 
auc = roc_auc_score(df_survived['Survived'], Y_pred) 
print('AUC: %.5f' % auc) 
Sensitivity = cm[0][0] / (cm[0][0] + cm[1][0]) # Sensitivity = TP / (TP + FN) 
Specificity = cm[1][1] / (cm[1][1] + cm[0][1]) # Specificity = TN / (TN + FN) 
Accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][0] + cm[0][1] + cm[1][1]) # Accuracy = TP + TN / (TP + TN + FP + FN) 
Error = (cm[1][0] + cm[0][1]) / (cm[0][0] + cm[1][0] + cm[0][0] + cm[1][0]) # Error = FP + FN / (TP + TN + FN + FP) 
print(Sensitivity) print(Specificity) print(Accuracy) print(Error)


