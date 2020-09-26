import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
# calculate LSE
def lse(V, W, H):
 return np.linalg.norm(V - np.matmul(W, H))
#UPDATE THE RULE

def update_rules(V, W, H):

 WtV = np.matmul(W.T, V)
 WtWH = np.matmul(np.matmul(W.T, W), H)

 H = H * WtV / WtWH

 VHt = np.matmul(V, H.T)
 WHHt = np.matmul(np.matmul(W, H), H.T)

 W = W * VHt / WHHt

 return W, H
def init_WH(k):
 # k denotes the number of clusters
 # Init W and H matrices with 0.1 value
 #W, H = np.full([V.shape[0], k], 0.1), np.full([k, V.shape[1]], 0.1)

 W, H = np.random.uniform(0, 1, (V.shape[0], k)), np.random.uniform(0, 1, (k, V.shape[1]))
 return W, H
#calculate I and J
def calc_IJ(W, H):

 H_norm = normalize(H.T, axis=0).T
 W_norm = normalize(W, axis=0).T

 H_mean = np.mean(H_norm)
 W_mean = np.mean(W_norm)

 I = [np.where(H_norm[i] >= H_mean)[0]for i in range(H_norm.shape[0])]
 J = [np.where(W_norm[i] >= W_mean)[0]for i in range(W_norm.shape[0])]

 idx = np.argsort([ix[0] for ix in I])

 I = [I[i] for i in idx]
 J = [J[j] for j in idx]

 return I, J
def Jaccard(I1, J1, I2, J2):
 Jacc = []
 for i in range (len(I1)):
 i1, i2, j1, j2 = set(I1[i]), set(I2[i]), set(J1[i]), set(J2[i])
 Jacc.append(((len(i1.intersection(i2)) / len(i1.union(i2))) + (len(j1.intersection(j2)) /
len(j1.union(j2))))/2)

 return Jacc
np.random.seed(0)
dataset_no = 2
max_iter = 10000
threshold = 1e-6

k = pd.read_csv('/Users/rathi/Downloads/data/1.vmatrix', delimiter='\t',
nrows=1,header=None).values[0][0]
df = pd.read_csv('/Users/rathi/Downloads/data/1.vmatrix', delimiter='\t', header = None, skiprows=[i
for i in range (k+1)], decimal =',')
GT = np.array(pd.read_csv('/Users/rathi/Downloads/data/1.vmatrix', sep="\t", nrows=k, skiprows=1,
decimal=",", header=None)) V = df.values.astype(np.float)
V = df.values.astype(np.float)
k = pd.read_csv('/Users/rathi/Downloads/data/2.vmatrix', delimiter='\t',
nrows=1,header=None).values[0][0]
df = pd.read_csv('/Users/rathi/Downloads/data/2.vmatrix', delimiter='\t', header = None, skiprows=[i
for i in range (k+1)], decimal =',')
GT = np.array(pd.read_csv('/Users/rathi/Downloads/data/2.vmatrix', sep="\t", nrows=k, skiprows=1,
decimal=",", header=None))
V = df.values.astype(np.float)
k = pd.read_csv('/Users/rathi/Downloads/data/3.vmatrix', delimiter='\t',
nrows=1,header=None).values[0][0]
df = pd.read_csv('/Users/rathi/Downloads/data/3.vmatrix', delimiter='\t', header = None, skiprows=[i
for i in range (k+1)], decimal =',')
GT = np.array(pd.read_csv('/Users/rathi/Downloads/data/3.vmatrix', sep="\t", nrows=k, skiprows=1,
decimal=",", header=None))V = df.values.astype(np.float)
V = df.values.astype(np.float)
k = pd.read_csv('/Users/rathi/Downloads/data/7.vmatrix', delimiter='\t',
nrows=1,header=None).values[0][0]
df = pd.read_csv('/Users/rathi/Downloads/data/7.vmatrix', delimiter='\t', header = None, skiprows=[i
for i in range (k+1)], decimal =',')
GT = np.array(pd.read_csv('/Users/rathi/Downloads/data/7.vmatrix', sep="\t", nrows=k, skiprows=1,
decimal=",", header=None))
V = df.values.astype(np.float)
k = pd.read_csv('/Users/rathi/Downloads/data/10.vmatrix', delimiter='\t',
nrows=1,header=None).values[0][0]
df = pd.read_csv('/Users/rathi/Downloads/data/10.vmatrix', delimiter='\t', header = None, skiprows=[i
for i in range (k+1)], decimal =',')
GT = np.array(pd.read_csv('/Users/rathi/Downloads/data/10.vmatrix', sep="\t", nrows=k, skiprows=1,
decimal=",", header=None))
V = df.values.astype(np.float)
k = pd.read_csv('/Users/rathi/Downloads/data/46.vmatrix', delimiter='\t',
nrows=1,header=None).values[0][0]
df = pd.read_csv('/Users/rathi/Downloads/data/46.vmatrix', delimiter='\t', header = None, skiprows=[i
for i in range (k+1)], decimal =',')
GT = np.array(pd.read_csv('/Users/rathi/Downloads/data/46.vmatrix', sep="\t", nrows=k, skiprows=1,
decimal=",", header=None))
V = df.values.astype(np.float)
# init w and h
W, H = init_WH(k)
#main loop updation
prev = lse(V, W, H)
print(f'After Initialization, LSE:[prev')
for i in range(max_iter):
 W, H = update_rules(V, W, H)
 diff = lse(V, W, H) - prev
 prev = lse(V, W, H)
 print(f'Iter: [i+1], LSE: [1se(V, W, H)], Differnace: [diff]')
 if diff > threshold:
 print('LSE differance is less than threshold')
 break

#ground truth
I, J = calc_IJ(W, H)
GT_I = GT[0:k, 1:GT[0, 0]+1]
GT_J = GT[0:k, GT[0, 0]+2:GT.shape[1]]
print(Jaccard(I, J, GT_I, GT_J))'''
