import numpy as np
import scipy.io as sio
import math
import matplotlib.pyplot as plt
from sklearn import svm

def gaussian_kernel(x1, x2, sigma):
	similarity = np.exp(-np.square(np.linalg.norm(x1-x2))/(2.0*(sigma**2)))
	return similarity

def dataset3_params(x_data, y_data, xval, yval):
	best_score = float('-inf')
	best_c = 0
	best_sigma = 0
	values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
	for C in values:
		for sigma in values:
			svc = svm.SVC(C=C, gamma=sigma)
			svc.fit(x_data, y_data.ravel())
			score = svc.score(xval, yval.ravel())
			if score > best_score:
				'''
				print '\n'
				print C
				print sigma
				print score
				print '\n'
				'''
				best_score = score
				best_c = C
				best_sigma = sigma
	return best_score, best_c, best_sigma

mat_contents = sio.loadmat('ex6data1.mat')
#print mat_contents
#print mat_contents
x_data = mat_contents['X']
y_data = mat_contents['y']
#y_data = y_data.ravel()
#print np.shape(y_data)
#print np.shape(x_data)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_data[np.where(y_data==1), 0], x_data[np.where(y_data==1), 1], marker='x', c='b', label='Positive')
ax.scatter(x_data[np.where(y_data==0), 0], x_data[np.where(y_data==0), 1], marker='o', c='r', label='Negative')
ax.legend()
#plt.show()
#plt.title('Profits distribution')
#xlabel('Population of City in 10,000s')
#ylabel('Profit in $10,000s')
svc1 = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
print svc1
svc1.fit(x_data, y_data.ravel())
print svc1.score(x_data, y_data.ravel())

svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
print svc2
svc2.fit(x_data, y_data.ravel())
print svc2.score(x_data, y_data.ravel())

svm1_confidence = svc1.decision_function(x_data)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_data[:, 0], x_data[:, 1], s=50, c=svm1_confidence, cmap='seismic')
ax.set_title('SVM (C=1) Decision Confidence')

svm2_confidence = svc2.decision_function(x_data)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_data[:, 0], x_data[:, 1], s=50, c=svm2_confidence, cmap='seismic')
ax.set_title('SVM (C=100) Decision Confidence')

#plt.show()
#plt.close()
x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussian_kernel(x1, x2, sigma)
print 'Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n' % (sigma, sim)

mat_contents = sio.loadmat('ex6data2.mat')
#print mat_contents
#print mat_contents
x_data = mat_contents['X']
y_data = mat_contents['y']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_data[np.where(y_data==1), 0], x_data[np.where(y_data==1), 1], marker='x', c='b', label='Positive')
ax.scatter(x_data[np.where(y_data==0), 0], x_data[np.where(y_data==0), 1], marker='o', c='r', label='Negative')
ax.legend()

svc = svm.SVC(C=100, gamma=10, probability=True)
svc.fit(x_data, y_data.ravel())
predictions = svc.predict_proba(x_data)[:,0]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(x_data[:, 0], x_data[:, 1], s=30, c=predictions, cmap='Reds')
#plt.show()

mat_contents = sio.loadmat('ex6data3.mat')
#print mat_contents
#print mat_contents
x_data = mat_contents['X']
y_data = mat_contents['y']
xval = mat_contents['Xval']
yval = mat_contents['yval']

score, C, sigma = dataset3_params(x_data, y_data, xval, yval)
print score
print C
print sigma

mat_contents = sio.loadmat('spamTrain.mat')
x_data = mat_contents['X']
y_data = mat_contents['y']
mat_contents = sio.loadmat('spamTest.mat')
x_test = mat_contents['Xtest']
y_test = mat_contents['ytest']

svc = svm.SVC()
svc.fit(x_data, y_data.ravel())
print 'Training Accuracy: %f\n' % (svc.score(x_data, y_data.ravel()) * 100)
print 'Test Accuracy: %f\n' % (svc.score(x_test, y_test.ravel()) * 100)



























