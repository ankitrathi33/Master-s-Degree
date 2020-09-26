from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
import pandas as pd
data_set = pd.read_csv('origin_1.csv')
columns = data_set.columns
A_columns = columns[:10]
B_columns = columns[10:20]
C_columns = columns[20:30]
A = data_set[A_columns]
B = data_set[B_columns]
C = data_set[C_columns]
test = A.append(B,sort=False, ignore_index = True)
def scenario_one(X1,X2):
	# Dataset Prepartion Stage
	new_cols = {x: y for x, y in zip(X2.columns, X1.columns)}
	new_X2 = X2.rename(columns=new_cols)
	X_train = X1.append(new_X2,sort=False, ignore_index = True)
	Y_train = [0]*len(X1.index)
	Y_train.extend([1]*len(X2.index))
	column_index = {v:k for k,v in enumerate(X_train.columns)}
	#Feature Selection Stage
	model = LogisticRegression(solver='lbfgs')
	rfe = RFE(model, 3)
	fit = rfe.fit(X_train, Y_train)
	#Trainset Prepartion Stage
	selected_features = []
	for is_true,col_name in zip(fit.support_,X_train.columns):
		if is_true:
			selected_features.append(col_name)
	x_train = X_train[selected_features]
	# learnig stage
	lr = LogisticRegression(solver='lbfgs')
	lr.fit(x_train, Y_train)
	# evaluation stage
	print('Validation Scenario 1 - Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(x_train, Y_train)))


def scenario_two(X1,X2):
	# Dataset Prepartion Stage
	new_cols = {x: y for x, y in zip(X2.columns, X1.columns)}
	new_X2 = X2.rename(columns=new_cols)
	X_train = X1.append(new_X2,sort=False, ignore_index = True)
	Y_train = [0]*len(X1.index)
	Y_train.extend([1]*len(X2.index))
	column_index = {v:k for k,v in enumerate(X_train.columns)}

	#Feature Selection Stage
	model = LogisticRegression(solver='lbfgs')
	rfe = RFE(model, 3)
	fit = rfe.fit(X_train, Y_train)
	
	selected_features = []
	for is_true,col_name in zip(fit.support_,X_train.columns):
		if is_true:
			selected_features.append(col_name)

	#Trainset Prepartion stage  Data Paritioning
	x_train = X_train[selected_features]
	X_train, X_test, y_train, y_test = train_test_split(x_train, Y_train, random_state=0)

	# learnig stage
	lr = LogisticRegression(solver='lbfgs')
	lr.fit(X_train, y_train)

	# evaluation stage
	print('Validation Scenario 2 - Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(X_test, y_test)))


def scenario_three(X1,X2):
	# Dataset Prepartion Stage
	new_cols = {x: y for x, y in zip(X2.columns, X1.columns)}
	new_X2 = X2.rename(columns=new_cols)
	X_train = X1.append(new_X2,sort=False, ignore_index = True)
	Y_train = [0]*len(X1.index)
	Y_train.extend([1]*len(X2.index))
	column_index = {v:k for k,v in enumerate(X_train.columns)}

	# Data Partioning
	X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=0)

	#Feature Selection Stage
	model = LogisticRegression(solver='lbfgs')
	rfe = RFE(model, 3)
	fit = rfe.fit(X_train, y_train)
	
	selected_features = []
	for is_true,col_name in zip(fit.support_,X_train.columns):
		if is_true:
			selected_features.append(col_name)

	#Trainset Prepartion stage  Data Paritioning
	X_test = X_test[selected_features]
	X_train = X_train[selected_features]

	# learnig stage
	lr = LogisticRegression(solver='lbfgs')
	lr.fit(X_train, y_train)

	# evaluation stage
	print('Validation Scenario 3 - Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(lr.score(X_test, y_test)))
print("A vs B")
scenario_one(A,B)
scenario_two(A,B)
scenario_three(A,B)
print("A vs C")
scenario_one(A,C)
scenario_two(A,C)
scenario_three(A,C)
print("B vs C")
scenario_one(B,C)
scenario_two(B,C)
scenario_three(B,C)
