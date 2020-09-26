import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
# importing the dataset 
data = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')
# Data exploration 
print(data.shape)
data.info()
data.columns

#visualization 
# systematic crisis 
fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(data['country'],hue=data['systemic_crisis'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=50)

# check for banking crisis 

systemic = data[['year','country', 'systemic_crisis', 'banking_crisis']]
systemic = systemic[(systemic['country'] == 'Central African Republic') | (systemic['country']=='Kenya') | (systemic['country']=='Zimbabwe') ]
plt.figure(figsize=(20,15))
count = 1

for country in systemic.country.unique():
    plt.subplot(len(systemic.country.unique()),1,count)
    subset = systemic[(systemic['country'] == country)]
    sns.lineplot(subset['year'],subset['systemic_crisis'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Systemic Crisis/Banking Crisis')
    plt.title(country)
    count+=1
    
# Exchange rate over the years 

plt.figure(figsize=(15,30))
count = 1
for country in data.country.unique():
    plt.subplot(len(data.country.unique()),1,count)
    count+=1
    sns.lineplot(data[data.country==country]['year'],data[data.country==country]['exch_usd'])
    plt.subplots_adjust(hspace=0.8)
    plt.xlabel('Years')
    plt.ylabel('Exchange Rates')
    plt.title(country)
    
# sovereign domestic debt default 

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(data['country'],hue=data['domestic_debt_in_default'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)

#Angola and Zimbabe 

sovereign = data[['year','country', 'domestic_debt_in_default', 'banking_crisis']]
sovereign = sovereign[(sovereign['country'] == 'Angola') | (sovereign['country']=='Zimbabwe') ]
plt.figure(figsize=(20,15))
count = 1

for country in sovereign.country.unique():
    plt.subplot(len(sovereign.country.unique()),1,count)
    subset = sovereign[(sovereign['country'] == country)]
    sns.lineplot(subset['year'],subset['domestic_debt_in_default'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Sovereign Domestic Debt Defaults/Banking Crisis')
    plt.title(country)
    count+=1
    
# sovereign external debt default 

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(data['country'],hue=data['sovereign_external_debt_default'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)

#We see only Central African Republic , Ivory Coast and Zimbabwe defaulting. Let's take a look at a few other things for these three countries in particular

sovereign_ext = data[['year','country', 'sovereign_external_debt_default', 'banking_crisis']]
sovereign_ext = sovereign_ext[(sovereign_ext['country'] == 'Central African Republic') | (sovereign_ext['country'] == 'Ivory Coast') | (sovereign_ext['country']=='Zimbabwe') ]
plt.figure(figsize=(20,15))
count = 1

for country in sovereign_ext.country.unique():
    plt.subplot(len(sovereign_ext.country.unique()),1,count)
    subset = sovereign_ext[(sovereign_ext['country'] == country)]
    sns.lineplot(subset['year'],subset['sovereign_external_debt_default'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Sovereign Ext Debt Defaults/Banking Crisis')
    plt.title(country)
    count+=1
    
# currency crisis 

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(data['country'],hue=data['currency_crises'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)

curr = data[['year','country', 'currency_crises', 'banking_crisis']]
curr = curr[(curr['country'] == 'Angola') | (curr['country'] == 'Zambia') | (curr['country']=='Zimbabwe') ]
curr = curr.replace(to_replace=2, value=1, regex=False)

plt.figure(figsize=(20,15))
count = 1

for country in curr.country.unique():
    plt.subplot(len(curr.country.unique()),1,count)
    subset = curr[(curr['country'] == country)]
    sns.lineplot(subset['year'],subset['currency_crises'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Currency Crisis/Banking Crisis')
    plt.title(country)
    count+=1
    
#inflation crisis 

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(data['country'],hue=data['inflation_crises'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)

#Most commonly used inflation indexes are the Consumer Price Index (CPI). Let's look at the Annual CPI for the three countries to see if we can derive any insights.
infla = data[['year','country', 'inflation_crises', 'inflation_annual_cpi', 'banking_crisis']]
infla = infla[(infla['country'] == 'Angola') | (infla['country'] == 'Zambia') | (infla['country']=='Zimbabwe') ]
infla = infla.replace(to_replace=2, value=1, regex=False)

plt.figure(figsize=(20,15))
count = 1

for country in infla.country.unique():
    plt.subplot(len(infla.country.unique()),1,count)
    subset = infla[(infla['country'] == country)]
    sns.lineplot(subset['year'],subset['inflation_crises'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Inflation Crisis/Banking Crisis')
    plt.title(country)
    count+=1

plt.figure(figsize=(20,15))
count = 1

for country in infla.country.unique():
    plt.subplot(len(infla.country.unique()),1,count)
    subset = infla[(infla['country'] == country)]
    sns.lineplot(subset[subset.country==country]['year'], subset[subset.country==country]['inflation_annual_cpi'])
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Annual CPI')
    plt.title(country)
    count+=1
    
#Logistic regression 

X = data[['systemic_crisis', 'exch_usd', 'domestic_debt_in_default',
       'sovereign_external_debt_default', 'gdp_weighted_default',
       'inflation_annual_cpi', 'independence', 'currency_crises',
       'inflation_crises']]
# Define the Y variable 
Y = data['banking_crisis']

## percentage of  crisis & no crisis

count_no_crisis = len(data[Y=='no_crisis'])
count_crisis = len(data[Y=='crisis'])
pct_of_no_crisis = count_no_crisis/(count_no_crisis+count_crisis)
print("percentage of no crisis is", pct_of_no_crisis*100)
pct_of_crisis = count_crisis/(count_no_crisis+count_crisis)
print("percentage of crisis", pct_of_crisis*100)

Y= pd.get_dummies(Y)
Y = Y.drop(['no_crisis'], axis = 1)

# Over-sampling using SMOTE
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_Y=os.fit_sample(X_train, Y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_Y= pd.DataFrame(data=os_data_Y,columns=['Y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of crisis in oversampled data",len(os_data_Y[os_data_Y['Y']==1]))
print("Number of no crisis",len(os_data_Y[os_data_Y['Y']==0]))
print("Proportion of no crisis data in oversampled data is ",len(os_data_Y[os_data_Y['Y']==0])/len(os_data_X))
print("Proportion of crisis data in oversampled data is ",len(os_data_Y[os_data_Y['Y']==1])/len(os_data_X))

#Implementing the model 

X=os_data_X
Y=os_data_Y
import statsmodels.api as sm
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())

#p-value for most of the variable are smaller so we remove them 

cols=['systemic_crisis', 'exch_usd','gdp_weighted_default','independence'] 
X=os_data_X[cols]
y=os_data_Y['Y']
logit_model=sm.Logit(Y,X)
result=logit_model.fit()
print(result.summary2())

from sklearn import linear_model
clf = linear_model.LogisticRegression()
clf.fit(X, Y)
clf.score(X,Y)
pd.DataFrame(zip(X.columns, np.transpose(clf.coef_)))
#model validation and evaluation 

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, Y_test))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test, Y_pred)
print(confusion_matrix)

#ROC curve 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(Y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(Y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#Simple neural network model 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')


df['banking_crisis'] = df['banking_crisis'].replace('crisis',np.nan)
df['banking_crisis'] = df['banking_crisis'].fillna(1)
df['banking_crisis'] = df['banking_crisis'].replace('no_crisis',np.nan)
df['banking_crisis'] = df['banking_crisis'].fillna(0)

# removing unneccesary data

df.drop(['cc3','country'], axis=1, inplace=True)

# scaling the data

df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['banking_crisis'] = df['banking_crisis']
df = df_scaled

# defining the input data, X, and the desired results, y 

X = df.loc[:,df.columns != 'banking_crisis']
y = df.loc[:, 'banking_crisis']

# breaking data into training data, validation data, and test data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2)

print(X.shape)

#Network architecture model 
model = Sequential()
model.add(Dense(11, input_dim=11, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the network

model.fit(X_train, y_train, epochs=200)

# scoring it on the data it trained on as well as test data

scores = model.evaluate(X_train, y_train)
print ("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print ("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

model = Sequential()
model.add(Dense(5, input_dim=11, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# training the network

model.fit(X_train, y_train, epochs=200)

# scoring it on the data it trained on as well as test data

scores = model.evaluate(X_train, y_train)
print ("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(X_test, y_test)
print ("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# plotting the confusion matrix

y_test_pred = model.predict_classes(X_test)
c_matrix = confusion_matrix(y_test,y_test_pred)
ax = sns.heatmap(c_matrix, annot=True, xticklabels=['No Banking Crisis','Banking Crisis'], yticklabels=['No Banking Crisis','Banking Crisis'], cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
   




