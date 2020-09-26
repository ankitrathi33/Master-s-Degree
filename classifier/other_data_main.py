from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import pandas as pd
import sys 

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pd.read_csv(url, names=names)
array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# print (X)
# print(X[0])
# sys.exit()

def scenario_one():
	global dataframe,names
	array = dataframe.values
	X_train = array[:,0:8]
	Y_train = array[:,8]
	#Feature Selection Stage
	model = LogisticRegression(solver='lbfgs',max_iter =10000)
	rfe = RFE(model, 3)
	fit = rfe.fit(X_train, Y_train)
	#Trainset Prepartion Stage
	selected_features = []
	for is_true,col_name in zip(fit.support_,names):
		if is_true:
			selected_features.append(col_name)
	x_train = dataframe[selected_features]
	x_train = x_train.values
	# learnig stage
	lr = LogisticRegression(solver='lbfgs',max_iter =10000)
	lr.fit(x_train, Y_train)
	# evaluation stage
	print('Validation Scenario 1 - Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(x_train, Y_train)))


def scenario_two():
	global dataframe,names
	array = dataframe.values
	X_train = array[:,0:8]
	Y_train = array[:,8]

	#Feature Selection Stage
	model = LogisticRegression(solver='lbfgs',max_iter =10000)
	rfe = RFE(model, 3)
	fit = rfe.fit(X_train, Y_train)
	
	selected_features = []
	for is_true,col_name in zip(fit.support_,names):
		if is_true:
			selected_features.append(col_name)

	#Trainset Prepartion stage  Data Paritioning
	x_train = dataframe[selected_features].values
	X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, random_state=0)

	# learnig stage
	lr = LogisticRegression(solver='lbfgs')
	lr.fit(X_train, y_train)

	# evaluation stage
	print('Validation Scenario 2 - Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(X_test, y_test)))


def scenario_three():
	# Dataset Prepartion Stage
	global dataframe,names
	array = dataframe.values
	X_train = array[:,0:8]
	Y_train = array[:,8]

	# Data Partioning
	X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=0)

	#Feature Selection Stage
	model = LogisticRegression(solver='lbfgs',max_iter=10000)
	rfe = RFE(model, 3)
	fit = rfe.fit(X_train, y_train)
	
	selected_features = []
	for index,is_true in enumerate(fit.support_):
		if is_true:
			selected_features.append(index)

	#Trainset Prepartion stage  Data Paritioning
	temp = []
	for item in X_train:
		new_arr = []		
		for feature in selected_features:
			new_arr.append(item[feature])
		temp.append(new_arr)
	X_train = temp

	temp = []
	for item in X_test:
		new_arr = []		
		for feature in selected_features:
			new_arr.append(item[feature])
		temp.append(new_arr)
	X_test = temp
		
	# X_test = dataframe[selected_features].values
	# X_train = dataframe[selected_features].values

	# learnig stage
	lr = LogisticRegression(solver='lbfgs')
	lr.fit(X_train, y_train)

	# evaluation stage
	print('Validation Scenario 3 - Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(X_test, y_test)))
scenario_one()
scenario_two()
scenario_three()
