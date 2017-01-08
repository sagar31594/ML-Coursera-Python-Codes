import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt

def feature_normalize(x_data):
	mu = np.mean(x_data, axis=0)
	sigma = np.std(x_data, axis=0)
	'''
	print np.mean(x_data[:, 0])
	print np.mean(x_data[:, 1])
	print np.std(x_data[:, 0])
	print np.std(x_data[:, 1])
	'''
	#print np.shape(mu)
	#print np.shape(sigma)
	#print 'sigma: ', sigma
	x_norm = (x_data-mu)/sigma
	return x_norm, mu, sigma

def pca(x_data):
	m, n = np.shape(x_data)
	covariance = (1.0/m)*np.dot(x_data.T, x_data)
	U, S, V = np.linalg.svd(covariance)
	#print np.shape(U)
	#print np.shape(S)
	#print np.shape(V)
	return U, S

def project_data(x_data, U, K):
	return np.dot(x_data, U[:, 0:K])

def recover_data(Z, U, K):
	return np.dot(Z, U[:, 0:K].T)

mat_contents = sio.loadmat('ex7data1.mat')
#print mat_contents
#print mat_contents
x_data = mat_contents['X']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_data[:, 0], x_data[:, 1], marker='o')
#plt.show()

print '\nRunning PCA on example dataset.\n\n'
x_norm, mu, sigma = feature_normalize(x_data)
#print mu
#print sigma
#raw_input()
U, S = pca(x_norm)
#fig, ax = plt.subplots(figsize=(12,8))
temp1 = mu + 1.5 * S[0] * U[:, 0]
temp2 = mu + 1.5 * S[1] * U[:, 1]
ax.plot([mu[0], temp1[0]], [mu[1], temp1[1]], c='k')
ax.plot([mu[0], temp2[0]], [mu[1], temp2[1]], c='k')
plt.show()
plt.clf()
plt.close()
#plt.plot([1,2], [3,4])
#plt.show()
print 'Top eigenvector: \n'
print ' U[:, 0] = ', U[:, 0]
print '\n(you should expect to see -0.707107 -0.707107)\n'

print '\nDimension reduction on example dataset.\n\n'
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_norm[:, 0], x_norm[:, 1], marker='o')
#plt.show()

K = 1
Z = project_data(x_norm, U, K)
print 'Projection of the first example: %f\n', Z[0]
print '\n(this value should be about 1.481274)\n\n'

x_rec  = recover_data(Z, U, K);
print 'Approximation of the first example: ', x_rec[0, :]
print '\n(this value should be about  -1.047419 -1.047419)\n\n'
#print x_rec
ax.scatter(x_rec[:, 0], x_rec[:, 1], marker='o', c='r')
#plt.show()
for j in range(np.shape(x_rec)[0]):
	ax.plot([x_norm[j, 0], x_rec[j, 0]], [x_norm[j, 1], x_rec[j, 1]], c='k', linestyle='--')
plt.show()












