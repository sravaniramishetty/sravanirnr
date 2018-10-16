k_range = range(1,10)
distortions = []

for i in k_range:
	kmeanModel = KMeans(n_clusters = i)
	kmeanModel.fit(X)
	distortions.append(sum(np.min(cdist(X,kmeanModel.cluster_centers_,'euclidean'),axis=1)) /X.shape[0])

fig1 = plt.figure()
ex = fig1.add_subplot(111)
ex.plot(k_range,distortions, 'b*-')
