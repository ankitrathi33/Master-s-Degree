import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances

dataset = pd.read_csv("/Users/rathi/Downloads/Data.csv",sep='\\t')
dataset.columns=[i.replace('"','') for i in dataset.columns ]
dataset['Age']=dataset['Age'].apply(lambda a: a.replace('"','')).astype(float)
dataset['Classification']=dataset['Classification'].apply(lambda a: a.replace('"','')).astype(float)


def cal_bic(ll, n, k):
    return -2 * ll + np.log(n) * k

#1 Data Preparation

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, log_loss
from sklearn.model_selection import train_test_split
from pprint import pprint

def main():

    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    pprint(y_test)
    THR = 0.5
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    preds = np.where(model.predict_proba(X_test)[:, 1] > THR, 1, 0)

    AU1 = pd.DataFrame(dataset)

    AU2 = pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                             precision_score(y_test, preds), roc_auc_score(y_test, preds)],
                       index=["accuracy", "recall", "precision", "roc_auc_score"])
    log_likelihood = -log_loss(y_test, preds)
    r, c = AU1.shape
    bic = cal_bic(log_likelihood, r, c - 1)
    print("BIC: \t\t", bic)
    print("LOG-LIKELIHOOD: \t\t", log_likelihood)
    print(AU2)
    pprint(y_test)
    pprint(preds)
   # exit(0)
        
    
def reliefF(X, y, **kwargs):
    if "k" not in kwargs.keys():
        k = 5
    else:
        k = kwargs["k"]
    n_samples, n_features = X.shape

    # calculating pairwise distance between instances
    distance = pairwise_distances(X, metric='manhattan')

    score = np.zeros(n_features)

    # the number of sampled instances is equal to the number of total instances
    for idx in range(n_samples):
        near_hit = []
        near_miss = dict()

        self_fea = X[idx, :]
        z = np.unique(y).tolist()

        stop_dict = dict()
        for label in z:
            stop_dict[label] = 0
        del z[z.index(y[idx])]

        q_dict = dict()
        q_label_idx = float(len(y[y == y[idx]]))/float(n_samples)

        for label in z:
            q_label_z = float(len(y[y == label]))/float(n_samples)
            q_dict[label] = q_label_z/(1-q_label_idx)
            near_miss[label] = []

        Dist_sort = []
        distance[idx, idx] = np.max(distance[idx, :])

        for i in range(n_samples):
            Dist_sort.append([distance[idx, i], int(i), y[i]])
        Dist_sort.sort(key=lambda x: x[0])

        for i in range(n_samples):
            # find k nearest hit points
            if Dist_sort[i][2] == y[idx]:
                if len(near_hit) < k:
                    near_hit.append(Dist_sort[i][1])
                elif len(near_hit) == k:
                    stop_dict[y[idx]] = 1
            else:
                # find k nearest miss points for each label
                if len(near_miss[Dist_sort[i][2]]) < k:
                    near_miss[Dist_sort[i][2]].append(Dist_sort[i][1])
                else:
                    if len(near_miss[Dist_sort[i][2]]) == k:
                        stop_dict[Dist_sort[i][2]] = 1
            stop = True
            for (key, value) in stop_dict.items():
                    if value != 1:
                        stop = False
            if stop:
                break

        # update reliefF score
        near_hit_term = np.zeros(n_features)
        for ele in near_hit:
            near_hit_term = np.array(abs(self_fea-X[ele, :]))+np.array(near_hit_term)

        near_miss_term = dict()
        for (label, miss_list) in near_miss.items():
            near_miss_term[label] = np.zeros(n_features)
            for ele in miss_list:
                near_miss_term[label] = np.array(abs(self_fea-X[ele, :]))+np.array(near_miss_term[label])
            score += near_miss_term[label]/(k*q_dict[label])
        score -= near_hit_term/k
    return score

#Wrapper 
    
from sklearn.feature_selection import RFE, RFECV
def main():
    
    X = dataset.iloc[:, :-1].values
    X_labels = dataset.iloc[:, :-1].columns.values
    y = dataset.iloc[:, -1].values
    # Creating a logistic model object
    model = LogisticRegression(solver='liblinear', max_iter=50)

    rfe_model = RFECV(model, cv=10)
    rfe_fit = rfe_model.fit(X, y)
    indexes = rfe_fit.get_support(indices=True)
    X_feat_labels = []
    for feature_list_index in indexes:
        # indexes.append()
        X_feat_labels.append(X_labels[feature_list_index])
    # Print the names of the most important features
    # for feature_list_index in rfe_fit.get_support(indices=True):
    #     print(X_labels[feature_list_index])
    X_feat = X[:, indexes]
    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=0)
    pprint(y_test)
    THR = 0.5
    # model = LogisticRegression(solver='lbfgs', max_iter=50)
    # model = LogisticRegression(solver='lbfgs', class_weight="balanced")
    model.fit(X_train, y_train)
    preds = np.where(model.predict_proba(X_test)[:, 1] > THR, 1, 0)
    AU = pd.DataFrame(X_feat, columns=X_feat_labels)
    AU2 = pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                             precision_score(y_test, preds), roc_auc_score(y_test, preds)],
                       index=["accuracy", "recall", "precision", "roc_auc_score"])
    log_likelihood = -log_loss(y_test, preds)
    r, c = AU.shape
    bic = cal_bic(log_likelihood, r, c - 1)
    print(AU)
    print("Best Features:\t\t", X_feat_labels)
    print("BIC:\t\t", bic)
    print(AU2)
    pprint(y_test)
    pprint(preds)
    
# correlation 

from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pointbiserialr, spearmanr

THR = 0.5

def main():
    X = dataset.iloc[:, :-1].values
    X_labels = dataset.iloc[:, :-1].columns.values
    y = dataset.iloc[:, -1].values

    correlation_coef = []
    indexes = []
    X_feat_labels = []
    for i in range(len(X_labels)):
        pb = pointbiserialr(X[:, i], y)
        if pb.correlation > 0 and pb.pvalue < THR:
            indexes.append(i)
            X_feat_labels.append(X_labels[i])
        correlation_coef.append(pb.correlation)
    X_feat = X[:, indexes]
    corr_matrix = pd.DataFrame(X_feat, columns=X_feat_labels).corr(method='spearman')
    sns.heatmap(corr_matrix)
    indexes_final = []
    X_feat_labels_final = []
    for i in indexes:
        coef, pvalue = spearmanr(X[:, i], y)
        # print(coef, pvalue)
        if pvalue < THR:
            X_feat_labels_final.append(X_labels[i])
            indexes_final.append(i)
    X_feat_final = X[:, indexes_final]
    X_train, X_test, y_train, y_test = train_test_split(X_feat_final, y, test_size=0.2, random_state=0)
    model = LogisticRegression(solver='liblinear', max_iter=50)
    # model = LogisticRegression(solver='lbfgs', class_weight="balanced")
    model.fit(X_train, y_train)
    preds = np.where(model.predict_proba(X_test)[:, 1] > THR, 1, 0)
    AU = pd.DataFrame(X_feat_final, columns=X_feat_labels_final)
    AU2 = pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                             precision_score(y_test, preds), roc_auc_score(y_test, preds)],
                       index=["accuracy", "recall", "precision", "roc_auc_score"])
    log_likelihood = -log_loss(y_test, preds)
    r, c = AU.shape
    bic = cal_bic(log_likelihood, r, c - 1)
    pprint(X_feat_labels_final)
    print("BIC:\t\t", bic)
    print(AU2)
    pprint(y_test)
    pprint(preds)
    plt.show()
    #exit(0)
        
if __name__ == '__main__':
    main()
    

