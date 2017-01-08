import numpy as np
import scipy.io as sio
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def linear_regression_cost_fn(theta, x_data, y_data, lambda_param):
	#print np.shape(x_data)
	#print x_data
	m = np.shape(x_data)[0]
	n = np.shape(x_data)[1]
	J = 0
	actual_theta = theta.reshape((n, 1))
	grad = np.zeros(np.shape(actual_theta))
	hTheta = x_data * actual_theta
	J = (1.0/(2*m))*np.sum(np.square(hTheta - y_data)) + (lambda_param / (2.0*m) * np.sum(np.power(actual_theta[1:], 2)))
	grad = (1.0/m)*np.transpose(np.transpose(hTheta - y_data) * x_data) + (lambda_param * 1.0 / m)*np.insert(actual_theta.reshape((n,1))[1:], 0, [0], axis=0)
	return J, np.asarray(grad).flatten()

def train_linear_reg(x_data, y_data, lambda_param):
	initial_theta = np.zeros(np.shape(x_data)[1])
	#print initial_theta
	#print initial_theta
	#raw_input()
	fmin = minimize(fun=linear_regression_cost_fn, x0=initial_theta, args=(x_data, y_data, lambda_param), method='TNC', jac=True, options={'maxiter': 200})
	return np.matrix(fmin.x.reshape((np.shape(x_data)[1], 1)))

def learning_curve(x_data, y_data, xval, yval, lambda_param):
	m = np.shape(x_data)[0]
	error_train = np.zeros((m, 1))
	error_val = np.zeros((m, 1))
	for i in range(1, m):
		theta_train = train_linear_reg(x_data[:i, :], y_data[:i, :], lambda_param)
		J_train, grad_train = linear_regression_cost_fn(theta_train, x_data[:i, :], y_data[:i, :], 0)
		error_train[i, 0] = J_train
		J_val, grad_val = linear_regression_cost_fn(theta_train, xval, yval, 0)
		error_val[i, 0] = J_val
	return error_train, error_val

def poly_features(x_data, p):
	x_poly = np.matrix(np.zeros((np.shape(x_data)[0], p)))
	#print np.shape(x_poly[:, 1])
	#print x_data[:, 0]
	#print np.shape(np.power(x_data[:, 0], 2))
	#raw_input()
	for i in range(p):
		x_poly[:, i] = np.power(x_data, i+1).reshape(np.shape(x_poly)[0], 1)
	return x_poly

def feature_normalize(x_poly):
	#print 'shape of x_poly: ', np.shape(x_poly)
	mu = np.mean(x_poly, axis=0)
	sigma = np.std(x_poly, axis=0)
	#print np.shape(mu)
	#print np.shape(sigma)
	#print 'sigma: ', sigma
	x_poly = (x_poly-mu)/sigma
	return x_poly, mu, sigma

def validation_curve(x_data, y_data, xval, yval):
	lambda_vec = np.array([0,0.001,0.003,0.01,0.03,0.1,0.3,1,3,10])
	lambda_vec = lambda_vec.reshape((np.size(lambda_vec), 1))
	error_train = np.zeros((np.shape(lambda_vec)[0], 1))
	error_val = np.zeros((np.shape(lambda_vec)[0], 1))
	for i in range(0, np.shape(lambda_vec)[0]):
		lambda_param = lambda_vec[i, 0]
		theta_train = train_linear_reg(x_data, y_data, lambda_param)
		J_train, grad_train = linear_regression_cost_fn(theta_train, x_data, y_data, 0)
		error_train[i, 0] = J_train
		J_val, grad_val = linear_regression_cost_fn(theta_train, xval, yval, 0)
		error_val[i, 0] = J_val
	return lambda_vec, error_train, error_val


mat_contents = sio.loadmat('ex5data1.mat')
#print mat_contents
print mat_contents
x_data = mat_contents['X']
y_data = mat_contents['y']
xval = mat_contents['Xval']
yval = mat_contents['yval']
xtest = mat_contents['Xtest']
ytest = mat_contents['ytest']
#print x_data
#print y_data
plt.plot(x_data, y_data, 'ro')
plt.axis([-60, 40, 0, 50])
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')
#plt.show()

constants = np.ones((np.shape(x_data)[0], 1))
x_data = np.matrix(np.append(constants, x_data, axis=1))
constants = np.ones((np.shape(xval)[0], 1))
xval = np.matrix(np.append(constants, xval, axis=1))
#print np.shape(x_data)[1]
y_data = np.matrix(y_data)
theta = np.ones((np.shape(x_data)[1], 1))
#print theta
J, grad = linear_regression_cost_fn(theta, x_data, y_data, 1)
#print x_data
#print np.shape(x_data)
print 'Cost at theta = [1; 1]: ', J
print 'This value should be about 303.993192\n'
print 'Gradient at theta = [1 ; 1]: ', grad[0], grad[1]
print 'This value should be about [-15.303016; 598.250744]\n'
lambda_param = 0
#print np.shape(x_data)
theta = train_linear_reg(x_data, y_data, lambda_param)
#print theta
#print x_data * theta
plt.plot(x_data[:, 1:], x_data * theta)
plt.show()

lambda_param = 0
error_train, error_val = learning_curve(x_data, y_data, xval, yval, lambda_param)
m = np.shape(x_data)[0]
plt.plot(np.arange(m)+1, error_train, 'b-', label='Training Error')
plt.plot(np.arange(m)+1, error_val, 'r-', label='Cross validation Error')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper right')
#plt.axis([0, 13, 0, 150])
plt.show()
print '# Training Examples\tTrain Error\tCross Validation Error'
for i in range(m):
	print '\t%d\t\t%f\t%f' % (i, error_train[i], error_val[i])
p = 8
x_poly = poly_features(x_data[:, 1], p)
print 'x_poly: '
print x_poly
x_poly, mu, sigma = feature_normalize(x_poly)
constants = np.ones((np.shape(x_poly)[0], 1))
x_poly = np.matrix(np.append(constants, x_poly, axis=1))

x_poly_test = poly_features(xtest, p)
print 'x_poly_test: '
print x_poly_test
print 'mu: '
print mu
print 'sigma: '
print sigma
x_poly_test = (x_poly_test-mu)/sigma#feature_normalize(x_poly_test)
constants = np.ones((np.shape(x_poly_test)[0], 1))
x_poly_test = np.matrix(np.append(constants, x_poly_test, axis=1))

x_poly_val = poly_features(xval[:, 1], p)
print 'x_poly_val: '
print x_poly_val
x_poly_val = (x_poly_val-mu)/sigma#feature_normalize(x_poly_val)
constants = np.ones((np.shape(x_poly_val)[0], 1))
x_poly_val = np.matrix(np.append(constants, x_poly_val, axis=1))

lambda_param = 0
theta = train_linear_reg(x_poly, y_data, lambda_param)
#print 'theta: ', theta
plt.plot(x_data[:,1], y_data, 'ro')
plt.xlabel('Change in water level')
plt.ylabel('Water flowing out of the dam')
plt.title('Polynomial Regression Fit (lambda = ' + str(lambda_param) + ')')
x = np.arange(np.min(x_data[:, 1])-15, np.max(x_data[:, 1])+25+0.05, 0.05)
#print np.min(x_data[:, 1])
#print np.max(x_data[:, 1])
x_other = poly_features(x, p)
x_other = (x_other-mu)/sigma
constants = np.ones((np.shape(x_other)[0], 1))
x_other = np.matrix(np.append(constants, x_other, axis=1))
#print 'np.shape1: ', np.shape(x)
#print 'np.shape1: ', np.shape(np.asarray(x_other * theta).flatten())
#print x_other * theta
plt.plot(x, np.asarray(x_other * theta).flatten(), linestyle='--')
plt.show()
#np.asarray(theta1).flatten()
error_train, error_val = learning_curve(x_poly, y_data, x_poly_val, yval, lambda_param)
'''
print x_poly
print y_data
print x_poly_val
print yval
'''
m = np.shape(x_poly)[0]
plt.plot(np.arange(m)+1, error_train, 'b-', label='Training Error')
plt.plot(np.arange(m)+1, error_val, 'r-', label='Cross validation Error')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Learning curve for linear regression')
plt.legend(loc='upper right')
#plt.axis([0, 13, 0, 150])
plt.show()
print '# Training Examples\tTrain Error\tCross Validation Error'
for i in range(m):
	print '\t%d\t\t%f\t%f' % (i, error_train[i], error_val[i])
lambda_vec, error_train, error_val = validation_curve(x_poly, y_data, x_poly_val, yval)
plt.plot(lambda_vec, error_train, 'b-', label='Training Error')
plt.plot(lambda_vec, error_val, 'r-', label='Cross validation Error')
plt.xlabel('lambda')
plt.ylabel('Error')
#plt.title('Learning curve for linear regression')
plt.legend(loc='upper right')
#plt.axis([0, 13, 0, 150])
plt.show()
print '# Lambda\tTrain Error\tCross Validation Error'
for i in range(np.size(lambda_vec)):
	print '\t%f\t\t%f\t%f' % (lambda_vec[i,0], error_train[i], error_val[i])

lambda_param = 3.0
theta_train = train_linear_reg(x_poly, y_data, lambda_param)
J_test, grad_test = linear_regression_cost_fn(theta_train, x_poly_test, ytest, 0)
print 'J_test'
print J_test

print 'Running 50 iterations of finding errors using randomly chosen examples.\n'
runs = 50
m = np.shape(x_data)[0]
error_train = np.zeros((m, 1))
error_val = np.zeros((m, 1))
total_val_examples = np.shape(x_poly_val)[0]
lambda_param = 0.01
for required_examples in range(1, m):
	error_train_total = 0
	error_val_total = 0
	print 'iteration: ', required_examples
	for i in range(runs):
		rnd_idx = np.random.randint(m, size=required_examples)
		rand_x_poly = x_poly[rnd_idx, :]
		rand_y = y_data[rnd_idx, :]
		theta_train = train_linear_reg(rand_x_poly, rand_y, lambda_param)
		J_train, grad_train = linear_regression_cost_fn(theta_train, rand_x_poly, rand_y, 0)
		error_train_total += J_train
		rnd_idx = np.random.randint(total_val_examples, size=required_examples)
		rand_x_poly_val = x_poly_val[rnd_idx, :]
		rand_yval = yval[rnd_idx, :]
		J_val, grad_val = linear_regression_cost_fn(theta_train, rand_x_poly_val, rand_yval, 0)
		error_val_total += J_val
	error_train[required_examples, 0] = error_train_total * 1.0 / runs
	error_val[required_examples, 0] = error_val_total * 1.0 / runs
plt.plot(np.arange(m)+1, error_train, 'b-', label='Training Error')
plt.plot(np.arange(m)+1, error_val, 'r-', label='Cross validation Error')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.title('Learning curve for linear regression with random examples')
plt.legend(loc='upper right')
plt.axis([0, 13, 0, 150])
plt.show()

