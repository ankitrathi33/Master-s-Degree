import numpy as np 
from sklearn.model_selection import train_test_split 
import pandas as pd  
import statsmodels.api as sm 
from sklearn.linear_model import LogisticRegression 
from sklearn import metrics 
from sklearn import model_selection 
 
 
 
df = pd.read_csv("/Users/rathi/Downloads/data.csv",sep='\\t')
seed = 1000
df.columns=[i.replace('"','') for i in df.columns ]
df['Age']=df['Age'].apply(lambda a: a.replace('"','')).astype(float)
df['Classification']=df['Classification'].apply(lambda a: a.replace('"','')).astype(float)


 
X = df.drop(["Classification"], axis=1) 
y = df['Classification'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, 
test_size=0.2, random_state=1000) 
thr = 0.5  
 
lm=sm.Logit(y_train,X_train) 
fit=lm.fit() 
print(fit.summary2()) 
y_test 
 
y_prediction = fit.predict(X_test) 
y_prediction  
y_class = np.where(y_prediction>thr,1,0) 
y_class 
acc_num= len(y_test) - np.count_nonzero(y_class-y_test)  
acc_num 
acc = acc_num/len(y_test) 
acc 
 
 
LR = LogisticRegression() 
fit = LR.fit(X_train,y_train) 
y_prediction = LR.predict(X_test) 
print('Accuracy of logistic regression classifier on test set: 
{:.2f}'.format(LR.score(X_test,y_test))) 
 
LRAnalysis(X_test_filt2=X__test_filt2, y_train=y_train,
y_test=y_test, y=y, X=X_filt_2, 
X_train_filt2=X__train_filt2) 
 
 
# part 2  
from scipy.stats import pointbiserialr 
from scipy.stats import spearmanr 
import matplotlib.pyplot as plt2 
import matplotlib.pyplot as plt 
import seaborn as sns 
 
pbsr = [] 
for i in range( X.shape[1] ): 
    pbsr.append( pointbiserialr(y, X.values[:,i]).correlation ) 
 
    print(pbsr) 
 
plt2.bar(range(9),pbsr )  
LABELS = list(X.columns)  
plt2.xticks(range(9), LABELS, rotation=20 ) 
plt2.show() 
X_filt = X[['Glucose','Insulin','HOMA','Resistin','MCP.1']] 
spearmanr_values = np.zeros((4,4)) 
for i in range(X_filt.shape[1]): 
    for j in range( X_filt.shape[1]): 
         spearmanr_values.append( 
spearmanr(X.values[:,i],X.values[:,j]).correlation ) )
        spearmanr_values[i][j] = 
spearmanr(X_filt.values[:,i],X_filt.values[:,j]).correlation

 
 
fig, ax = plt.subplots() 
plt.colorbar(plt.pcolor(spearmanr_values)) 
ax.set_xticks(np.arange(X_filt.shape[1])) 
ax.set_yticks(np.arange(X_filt.shape[1])) 
 
ax.set_xticklabels(list(X_filt.columns)) 
ax.set_yticklabels(list(X_filt.columns))  
plt.show() 
 
X_filt_2 = X_filt.drop(["Insulin"], axis=1) 
X__train_filt2 = X_train[['Glucose','Insulin','Resistin', 'HOMA']] 
X__test_filt2 = X_test[['Glucose','Insulin','Resistin', 'HOMA']] 

lm_filt2=sm.Logit(y,X_filt_2)
result=lm_filt2.fit() 
print(result.summary2()) 
LR_filt2 = LogisticRegression() LR_filt2.fit(X__train_filt2, y_train) 
y_prediction = LR_filt2.predict(X__test_filt2) 
print('Accuracy of logistic regression classifier on test set: 
{:.2f}'.format(LR_filt2.score(X__test_filt2, y_test))) 
 
 
LRAnalysis(X_test_filt2=X__test_filt2, y_train=y_train, 
y_test=y_test, y=y, X=X_filt_2, 
X_train_filt2=X__train_filt2) 
 
def LRAnalysis(y, X, y_train, y_test, X_train_filt2, 
X_test_filt2): 
    lm_filt2=sm.Logit(y_train,X_train_filt2) 
    fit=lm_filt2.fit() 
    print(fit.summary2()) 
 
    y_prediction = fit.predict(X_test_filt2) 
    y_prediction  
    y_class = np.where(y_prediction>thr,1,0) 
    y_class 
    acc_num= len(y_test) - np.count_nonzero(y_class - y_test)  
    acc_num 
    acc = acc_num/len(y_test) 
    acc

  
### RelieF 

#pip install skrebate
from skrebate import ReliefF 
import numpy as np 
from sklearn import datasets 
import pandas as pd 
 
train = pd.concat([X_train, y_train], axis=1, sort=False) 
train_samples_50 = train.sample(50) 
train_samples_50_X, train_samples_50_y = np.split(train_samples_50, 
[9], axis=1) 
feat_sel = ReliefF(n_neighbors=1, n_features_to_select=5 ) 
ReliefF_fit = feat_sel.fit(train_samples_50_X.values, 
train_samples_50_y.values.reshape(50)) 
 
 
ReliefF_fit.feature_imp_ 
plt2.bar(range(9),ReliefF_fit.feature_imp_ ) 
LABELS = list(X.columns) 
 
plt2.xticks(range(9), LABELS, rotation=20 ) 
plt2.show() 
 
impfeat_idx = ReliefF_fit.feature_imp_.argsort()[-4:][::-1] 
 
X_train_filt_relieff = X_train.iloc[:,impfeat_idx]
X_test_filt2_relieff = X_test.iloc[:,impfeat_idx] 
 
LRAnalysis(X_test_filt2=X_test_filt2_relieff, 
y_train=y_train, y_test=y_test, y=y, X=X, 
X_train_filt2=X_train_filt2_relieff) 


#  wrapper 
 
from sklearn.feature_selection import RFECV 
 
LR = LogisticRegression() 
 
selector = RFECV(LR, step=1, cv=10, min_features_to_select=9-5, 
scoring="roc_auc") 

selector = selector.fit(X, y) 
selector.support_ 
selector.ranking_ 
 
X_train_filt2_RFECV = X_train.iloc[:,selector.support_] 
X_test_filt2_RFECV = X_test.iloc[:,selector.support_] 
 
LRAnalysis(X_test_filt2=X_test_filt2_RFECV, 
y_train=y_train, y_test=y_test, y=y, X=X, 
X_train_filt2=X_train_filt2_RFECV) 
 

import feather as ft
from openpyxl import load_workbook
from openpyxl.styles import Font

df = [X_train, y_train, X_test, y_test] 
path1 = 'X_train.feather' 
feather.write_dfframe(X_train, path)
path2 = 'X_test.feather' 
feather.write_dfframe(X_test, path) 
path3 = 'y_test.feather' 
feather.write_dfframe(pd.dfFrame(y_test), path)
path4 = 'y_train.feather' 
feather.write_dfframe(pd.dfFrame(y_train), path)  
pd.dfFrame(y_train)
  
import matplotlib.pyplot as plt 
import numpy as np 
X.columns 
methods=['full','RFE','LASSO','Corr','Relief']   
a = np.array([[1,1,1,1,1,1,1,1,1],[0,0,1,1,0,0,0,1,0],[1,1,0,1,1,0,0,0,0],[
0,1,1,0,1,0,0,1,0],[0,0,1,0,0,0,0,0,0]]) 
a 
fig, ax = plt.subplots() 
plt.imshow(a, cmap='copper', interpolation='nearest') 
plt.colorbar() 
ax.set_xticks(np.arange(len(X.columns))) 
ax.set_yticks(np.arange(len(methods))) 
ax.set_xticklabels(X.columns) 
ax.set_yticklabels(methods) 
plt.show() 
