import numpy as np
import scipy.io as sio
import math
from scipy.optimize import minimize

NUM_LABELS = 10
INPUT_LAYER_SIZE = 400
HIDDEN_LAYER_SIZE = 25

def sigmoid(z):
	return 1.0/(1 + np.power(math.e, -z))

def predict(theta1, theta2, x_data):
	m = np.shape(x_data)[0]
	n = np.shape(x_data)[1]
	p = np.zeros((m, 1))

	a1 = x_data
	z2 = theta1 * np.transpose(a1)
	a2 = sigmoid(z2);
	a2 = np.transpose(a2)
	
	constants = np.ones((np.shape(a2)[0], 1))
	a2 = np.matrix(np.append(constants, a2, axis=1))
	z3 = theta2 * np.transpose(a2)
	a3 = sigmoid(z3)
	#hTheta = sigmoid(x_data * theta.reshape((n,1)))
	#hTheta = sigmoid(all_theta.reshape(NUM_LABELS, n) * np.transpose(x_data))
	#p = np.round(hTheta)
	p = np.transpose(np.argmax(a3, axis=0) + 1)
	#print np.shape(hTheta)
	return p

mat_contents = sio.loadmat('ex3data1.mat')
#print mat_contents
x_data = mat_contents['X']
constants = np.ones((np.shape(x_data)[0], 1))
x_data = np.matrix(np.append(constants, x_data, axis=1))
#print x_data
#print np.shape(x_data)
y_data = np.matrix(mat_contents['y'])
#print np.shape(y_data)
weights = sio.loadmat('ex3weights.mat')
#print weights
theta1 = weights['Theta1']
theta2 = weights['Theta2']
print np.shape(theta1)
print np.shape(theta2)

p = predict(theta1, theta2, x_data)
accuracy = (y_data[np.where(y_data==p)].size * 1.0 / y_data.size) * 100
print 'accuracy is ', accuracy

