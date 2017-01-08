import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt

def find_closest_centroids(x_data, centroids):
	K = np.shape(initial_centroids)[0]
	m = np.shape(x_data)[0]
	idx = np.zeros((m, 1))
	for i in range(m):
		idx[i, 0] = np.argmin(np.linalg.norm(x_data[i, :] - centroids, axis=1))
	return idx

def compute_centroids(x_data, idx, K):
	n = np.shape(x_data)[1]
	centroids = np.zeros((K, n))
	for i in range(K):
		centroids[i, :] = (1.0 * np.sum(x_data[np.where(idx[:, 0]==i)], axis=0)) / np.size(np.where(idx[:, 0]==i))
	return centroids

def run_kmeans(x_data, initial_centroids, max_iters, plotprogress=False):
	m, n = np.shape(x_data)
	K = np.shape(initial_centroids)[0]
	centroids = initial_centroids
	previous_centroids = centroids
	idx = np.zeros((m, 1))
	if plotprogress:
		fig, ax = plt.subplots(figsize=(12,8))
		ax.scatter(x_data[:, 0], x_data[:, 1], marker='o', c=idx)
	#plt.show()
	for i in range(max_iters):
		print 'K-Means iteration %d/%d...\n', i, max_iters
		idx = find_closest_centroids(x_data, centroids)

		if plotprogress:
			ax.scatter(x_data[:, 0], x_data[:, 1], marker='o', c=idx)
			ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='b', s=200)
			for j in range(np.shape(centroids)[0]):
				ax.plot([centroids[j, 0], previous_centroids[j, 0]], [centroids[j, 1], previous_centroids[j, 1]], c='r')
			previous_centroids = centroids
		centroids = compute_centroids(x_data, idx, K)
	if plotprogress:
		plt.show()
		#plt.clf()
	return centroids, idx

def init_centroids(x_data, K):
	m, n = np.shape(x_data)
	centroids = np.zeros((K, n))
	randidx = np.random.randint(0, m, K)
	for i in range(K):
		centroids[i, :] = x_data[randidx[i], :]
	return centroids

mat_contents = sio.loadmat('ex7data2.mat')
#print mat_contents
#print mat_contents
x_data = mat_contents['X']
K = 3
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroids(x_data, initial_centroids)
print idx
print 'Closest centroids for the first 3 examples: \n'
print idx[0:3, 0]
print '\n(the closest centroids should be 0, 2, 1 respectively)\n'

centroids = compute_centroids(x_data, idx, K)
print 'Centroids computed after initial finding of closest centroids: \n'
print centroids
print'\n(the centroids should be\n'
print'   [ 2.428301 3.157924 ]\n'
print'   [ 5.813503 2.633656 ]\n'
print'   [ 7.119387 3.616684 ]\n\n'
max_iters = 10
print initial_centroids
run_kmeans(x_data, initial_centroids, max_iters, True)
print '\nK-Means Done.\n\n'

print '\nRunning K-Means clustering on pixels from an image.\n\n'
mat_contents = sio.loadmat('bird_small.mat')
#print mat_contents
#print mat_contents
A = mat_contents['A']
A = A / 255.0
x_data = np.reshape(A, (np.shape(A)[0] * np.shape(A)[1], np.shape(A)[2]))
K = 16
max_iters = 10
initial_centroids = init_centroids(x_data, K)
centroids, idx = run_kmeans(x_data, initial_centroids, max_iters)
print '\nApplying K-Means to compress an image.\n\n'
idx = find_closest_centroids(x_data, centroids)
x_recovered = centroids[idx.astype(int),:]
x_recovered = np.reshape(x_recovered, (np.shape(A)[0], np.shape(A)[1], np.shape(A)[2]))
#plt.close()
#print x_recovered
fig, ax = plt.subplots(figsize=(12,8))
ax.imshow(x_recovered)
#plt.imshow(x_recovered)
plt.show()









