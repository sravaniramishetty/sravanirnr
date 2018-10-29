import matplotlib.pyplot as plt
x = [1,2,3]
y = [2,4,1]
plt.plot(x,y)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.show()

# Euclidean Distance Calculator
def dist(a,b,ax=1) :
	return np.linalg.norm(a-b, axis = ax)

# Number of clusters
k = 4
C_x = np.random.randint(0,np.max(X)-20,size = k)
C_y = np.random.randint(0,np.max(X)-20,size = k)
C = np.array(list(zip(C_x,C_y)), dtype = np.float32)
print(C)

#plotting along with the centroids
ax1.scatter(f1,f2,c='#050505',s=7,label = 'first')
ax1.scatter(C_x,C_y,marker = '*',s=200,c='g',label = 'second')
#plt.legend(loc = 'upper left')
plt.title('fig')
plt.show()
