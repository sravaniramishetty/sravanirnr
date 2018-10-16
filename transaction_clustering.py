#%matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.spatial.distance import cdist

plt.rcParams['figure.figsize']= (16,9)
plt.style.use('ggplot')
#X,y = make_blobs(n_samples = 800, n_features = 3, centers = 4)




#importing the dataset
data = pd.read_csv('income_transactions.csv')
print(data.shape)
data.head()

#Getting the values and plotting

f1 = data['age'].values
f3 = data['income'].values
f2 = data['No_of_transactions'].values
Xone = np.array(list(zip(f1)))
Yone = np.array(list(zip(f2)))
Zone = np.array(list(zip(f3)))
X = np.array(list(zip(f1,f2,f3)))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.show()

def dist(a,b,ax = 1):
	return np.linalg.norm(a-b, axis = ax)

# Number of clusters
k = 3
C_x = np.random.randint(np.min(Xone),np.max(Xone),size = k)
C_y = np.random.randint(np.min(Yone),np.max(Yone),size = k)
C_z = np.random.randint(np.min(Zone),np.max(Zone),size = k)
C = np.array(list(zip(C_x,C_y,C_z)), dtype = np.float32)
print(C)

#K-means
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
error = dist(C,C_old,None)

while error!=0:
	for i in range(len(X)):
		distances = dist(X[i], C)
		cluster = np.argmin(distances)
		clusters[i] = cluster
	C_old = deepcopy(C)
	for i in range(k):
		points = [X[j] for j in range(len(X)) if clusters[j] == i]
		C[i] = np.mean(points , axis =0)
	error = dist(C,C_old,None)

colors = ['r','g','b','y','c','m']
#fig, ax = plt.subplots()
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

for i in range(k):
	points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
	ax.scatter(points[:,0], points[:,1],points[:,2],s = 7 , c = colors[i])

ax.scatter(C[:,0],C[:,1],C[:,2],s =200,marker = '*',c='#050505')
plt.show()


		
k_range = range(1,10)
distortions = []

for i in k_range:
	kmeanModel = KMeans(n_clusters = i)
	kmeanModel.fit(X)
	distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean'),axis=1)) /X.shape[0])

fig1 = plt.figure()
ex = fig1.add_subplot(111)
ex.plot(k_range,distortions, 'b*-')

plt.grid(True)
plt.ylim([0,45])
plt.xlabel('Number of clusters')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
plt.show()





