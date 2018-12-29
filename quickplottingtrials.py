import numpy as np
import matplotlib.pyplot as plt

spikes=np.load('/media/yomnaj/New Volume/Bio_Assignment_2/electrode1thresh35spikes.txt.npy')

# for x in spikes:


#     plt.plot(x)

# plt.show()


from sklearn.decomposition import PCA

# pca = PCA(n_components=2)
# principalComponents = pca.fit_transform(spikes[0])
#print(principalComponents)

# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)

# ax.grid()
# plt.show()

# from matplotlib.mlab import PCA
# result=  PCA(spikes[0])
print(spikes.shape)
pca_elec1 = PCA(n_components=2)
electrode1_decomposed_spike_values= pca_elec1.fit_transform(spikes[0])
"""
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['target'] == target
ax.scatter(electrode1_decomposed_spike_values[:,0],electrode1_decomposed_spike_values[:,1])
#ax.legend(targets)
ax.grid()
#ax.show()
plt.show()
"""
from sklearn.cluster import KMeans  
kmeans = KMeans(n_clusters=2)  
kmeans.fit(electrode1_decomposed_spike_values)
print(kmeans.cluster_centers_)  
plt.scatter(electrode1_decomposed_spike_values[:,0],electrode1_decomposed_spike_values[:,1], c=kmeans.labels_, cmap='rainbow')  
plt.show()

