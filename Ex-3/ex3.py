import numpy as np
import scipy.io as sio
import math
from scipy.optimize import minimize

NUM_LABELS = 10

def sigmoid(z):
	return 1.0/(1 + np.power(math.e, -z))

def grad(theta, x_data, y_data, lambda_param):
	#print np.shape(x_data)
	#print np.shape(theta)
	m = np.shape(y_data)[0]
	n = np.shape(x_data)[1]
	hTheta = sigmoid(x_data * theta.reshape((n,1)))
	grad = (1.0/m)*np.transpose(np.transpose(hTheta - y_data) * x_data) + (lambda_param * 1.0 / m)*np.insert(theta.reshape((n,1))[1:], 0, [0], axis=0)
	#print grad
	#print np.squeeze(np.asarray(grad))
	return np.squeeze(np.asarray(grad))

def computeCost(theta, x_data, y_data, lambda_param):
	J = 0
	m = np.shape(y_data)[0]
	n = np.shape(x_data)[1]
	#hTheta = np.transpose(np.dot(np.transpose(theta), np.transpose(x_data)))
	#hTheta = np.transpose(np.transpose(theta) * np.transpose(x_data))
	#print np.shape(x_data)
	#print np.shape(theta)
	hTheta = sigmoid(x_data * theta.reshape((n,1)))
	#print hTheta
	#print (1.0/(2*m))
	#print 'np.shape of hTheta: ', np.shape(hTheta)
	#print 'np.shape of y_data: ', np.shape(y_data)
	#print hTheta
	#print y_data
	#print np.transpose(y_data)
	#print hTheta - y_data
	#print np.sum((hTheta - y_data)**2)
	#J = (1.0/(2*m))*np.sum(np.square(hTheta - y_data))
	J = (1.0/m)*(np.transpose(-y_data)*np.log(hTheta) - np.transpose(1-y_data)*np.log(1-hTheta))
	J = J.item(0) + (lambda_param / (2.0*m) * np.sum(theta[1:]**2))
	return J

def one_vs_all(x_data, y_data, lambda_param):
	m = np.shape(x_data)[0]
	n = np.shape(x_data)[1]
	all_theta = np.zeros((NUM_LABELS, n))
	for c in range(1, NUM_LABELS + 1):
		theta = np.zeros(n)
		class_specific_y_data = np.transpose(np.matrix([1 if elem == c else 0 for elem in y_data]))
		fmin = minimize(fun=computeCost, x0=theta, args=(x_data, class_specific_y_data, lambda_param), method='TNC', jac=grad)
		all_theta[c-1,:] = fmin.x
	return all_theta

def predict(all_theta, x_data):
	m = np.shape(x_data)[0]
	n = np.shape(x_data)[1]
	p = np.zeros((m, 1))
	#hTheta = sigmoid(x_data * theta.reshape((n,1)))
	hTheta = sigmoid(all_theta.reshape(NUM_LABELS, n) * np.transpose(x_data))
	#p = np.round(hTheta)
	p = np.transpose(np.argmax(hTheta, axis=0) + 1)
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
n = np.shape(x_data)[1]
lambda_param = 0.1
all_theta = one_vs_all(x_data, y_data, lambda_param)

#print theta
#print computeCost(theta, x_data, y_data, lambda_param)
#print grad(theta, x_data, y_data, lambda_param)
#raw_input()
#print np.shape(theta)
#theta = opt.fmin_bfgs(computeCost, theta, fprime=grad, args=(x_data, y_data, lambda_param))
#theta = gradient_descent(x_data, y_data, theta, alpha, iterations, m)
#print 'theta is ', theta
#prob = np.dot(np.array([1, 45, 85]), theta)
#prob = sigmoid(prob)
#print prob
p = predict(all_theta, x_data)
accuracy = (y_data[np.where(y_data==p)].size * 1.0 / y_data.size) * 100
print 'accuracy is ', accuracy

