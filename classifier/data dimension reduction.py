from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
from pandas import DataFrame
import pandas as pd
import numpy as np
#Generating 2-class artificial classification dataset
X,y = make_blobs(n_samples=100, centers = 2, cluster_std = 1,  n_features=3)
dataset = X,y
X,y[:10]
f = pd.DataFrame(data=X)
f['y'] = y
f.head()


# In[ ]:


#scatter plot
data1 = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = data1.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()


# # TASK 2

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
features = ['0','1','2' 'y' ]
df = f
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(f)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf)


# # TASK 3

# In[ ]:


from sklearn.model_selection import train_test_split  
from sklearn.feature_selection import VarianceThreshold
d1=np.array(f.get(['0','1','y']))
y1 = np.array(f.get('y'))
y1.shape
data1 = pd.read_csv("dataR2.csv", nrows=40000)  
df.shape
train_features, test_features, train_labels, test_labels=train_test_split(  
    f.drop(labels=['y'], axis=1),
    f['y'],
    test_size=0.2,
    random_state=41)
constant_filter = VarianceThreshold(threshold=0)  
constant_filter.fit(train_features)
len(train_features.columns[constant_filter.get_support()])


# # TASK 4

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Split our data
train, test, train_labels, test_labels = train_test_split(f,y1,test_size=0.33,random_state=42)
#Initialize our classifier
gnb = GaussianNB()
#Train our classifier
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)
#Evaluate accuracy
print(accuracy_score(test_labels, preds))


# # TASK 5

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(f, y1, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# # TASK 6

# In[ ]:


scaler = StandardScaler()
#test_size: To know the original data whiich is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(f, y1, test_size=1/7.0, random_state=0)
#Fit on training set only.
scaler.fit(train_img)
#Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
#Make an instance of the Model
pca = PCA(.95)
pca.fit(train_img)
train_img.shape


# # 6B

# In[ ]:


#6b
##Remove constant features from training and test sets
train_features = constant_filter.transform(train_features)  
test_features = constant_filter.transform(test_features)

train_features.shape


# # 8

# In[ ]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test))) 


# # REAL DATA

# # TASK 1

# In[ ]:


import pandas as pd
import numpy as np
data1 = pd.read_csv("dataR2.csv")

data1.head()
#df = data1


# # TASK 2

# In[ ]:


features = ['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1','Classification']
df = data1
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data1)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
print(principalDf)


# # TASK 3

# In[ ]:


d1=np.array(data1.get(['Age', 'BMI', 'Glucose', 'Insulin', 'HOMA', 'Leptin', 'Adiponectin', 'Resistin', 'MCP.1']))
y1 = np.array(data1.get('Classification'))
y1.shape
data1 = pd.read_csv("dataR2.csv", nrows=40000)  
data1.shape
train_features, test_features, train_labels, test_labels=train_test_split(  
    data1.drop(labels=['Age'], axis=1),
    data1['Age'],
    test_size=0.2,
    random_state=41)
constant_filter = VarianceThreshold(threshold=0)  
constant_filter.fit(train_features)
len(train_features.columns[constant_filter.get_support()])



# # TASK 4

# In[ ]:


#Split our data
train, test, train_labels, test_labels = train_test_split(d1,y1,test_size=0.33,random_state=42)
#Initialize our classifier
gnb = GaussianNB()
#Train our classifier
model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)
#Evaluate accuracy
print(accuracy_score(test_labels, preds))


# # TASK 5

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(d1, y1, random_state=0)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test, y_test)))


# # TASK 6A

# In[ ]:


scaler = StandardScaler()
#test_size: To know the original data whiich is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(d1, y1, test_size=1/7.0, random_state=0)
#Fit on training set only.
scaler.fit(train_img)
#Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)
#Make an instance of the Model
pca = PCA(.95)
pca.fit(train_img)
train_img.shape


# In[ ]:


#6b
##Remove constant features from training and test sets
train_features = constant_filter.transform(train_features)  
test_features = constant_filter.transform(test_features)

train_features.shape


# # TASK 8

# In[ ]:


svm = SVC()
svm.fit(X_train,y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

