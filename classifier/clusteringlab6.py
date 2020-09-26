import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
import matplotlib.ticker as plticker 
import csv import math
### Artificial Data ###A Sample of dataset 
mean_class_1 = [-10, 10] covariance_class_1 = [[4, 0], [0, 4]]  # diagonal covariance-class 1
mean_class_2 = [5,20] covariance_class_2 = [[4, 0], [0,4]]  # diagonal covariance-class 2
mean_class_3 = [50,50] covariance_class_3 = [[4, 0], [0,4]]  # diagonal covariance-class 2
class_1=np.random.multivariate_normal(mean_class_1, covariance_class_1, 100) 
class_2=np.random.multivariate_normal(mean_class_2, covariance_class_2, 100) 
class_3=np.random.multivariate_normal(mean_class_3, covariance_class_3, 100) 
dataset_class_one=pd.DataFrame({'X1':class_1[:,0],'X2':class_1[:,1]}) 
dataset_class_two=pd.DataFrame({'X1':class_2[:,0],'X2':class_2[:,1]}) 
#dataset_class_three=pd.DataFrame({'X1':class_3[:,0],'X2':class_3[:,1]}) 
dataset=dataset_class_one.append(dataset_class_two) 
#dataset = dataset.append(dataset_class_three)
dataset = dataset.iloc[:, :].values
#####
def calculate_elbow(k, matrix_X, centers): 
    sum_clusters=0 vector_distance =0; 
    for k in range(k):
centroid = centers[k]
for i in range(len(matrix_X[k])):
for j in range(2):
vector_distance += (matrix_X[k][i][j] - 
centroid[j])**2 
sum_clusters+=(vector_distance**1/2)**2 vector_distance = 0
print(sum_clusters) return sum_clusters 
def calculate_coordinates(n, data): 
    x = [] y = []
for number_of_cluster in range(1, n, 1):
Kmean = KMeans(n_clusters=number_of_cluster) 
Kmean.fit(data)
centers = Kmean.cluster_centers_ cluster_number = Kmean.labels_
matrix_X = np.array([[]]*number_of_cluster) matrix_X = matrix_X.tolist() 
for i in range(len(data)):
matrix_X[cluster_number[i]].append(data[i])
x.append(number_of_cluster)
y.append(calculate_elbow(number_of_cluster, matrix_X, centers))
	#	y[0] = 50000
return x, y, y[0]
def draw_graph_of_elbow(data):
#    plt.clf()
x_axis, y_axis, limit = calculate_coordinates(10, data) 
plt.ylim(0, math.ceil(limit)) 
plt.xlim(1, ) 
plt.plot(x_axis, y_axis, '-ok', color='black', alpha=0.5) 
plt.xticks(np.arange(min(x_axis), max(x_axis)+1, 1.0)) 
plt.title("Elbow point") plt.xlabel("Total number of clusters ") 
plt.ylabel("W") plt.show()
### Artificial Data draw_graph_of_elbow(dataset)
### Real Data real_data = pd.read_csv('yeast.csv') real_data = real_data.iloc[1:, :].values list_of_real_data = []
def split_data(): 
    for index in range(real_data.shape[1]-1): 
        dataset_real_data=pd.DataFrame({'X1':real_data[:, index],'X2':real_data[:, index+1]}) 
        list_of_real_data.append(dataset_real_data) 
        draw_graph_of_elbow(dataset_real_data.iloc[:, :].values) 
        split_data()


