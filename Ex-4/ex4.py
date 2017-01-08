import numpy as np
import scipy.io as sio
import math
from scipy.optimize import minimize

NUM_LABELS = 10
INPUT_LAYER_SIZE = 400
HIDDEN_LAYER_SIZE = 25
COUNTER = 0
def random_initialize_weights(l_in, l_out):
	w = np.zeros((l_out, l_in+1))
	epsilon_init = 0.12
	w = np.matrix(np.random.rand(l_out, l_in+1)) * 2 * epsilon_init - epsilon_init
	return w

def sigmoid(z):
	return 1.0/(1 + np.power(math.e, -z))

def sigmoid_gradient(z):
	g = np.zeros(np.shape(z))
	g = np.multiply(sigmoid(z), (1-sigmoid(z)))
	return g

def nn_cost_function(nn_params, x_data, y_data, lambda_param):
	global COUNTER
	COUNTER += 1
	print COUNTER
	theta1 = np.reshape(nn_params[0:HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE + 1)], (HIDDEN_LAYER_SIZE, INPUT_LAYER_SIZE+1))
	theta2 = np.reshape(nn_params[HIDDEN_LAYER_SIZE*(INPUT_LAYER_SIZE + 1):], (NUM_LABELS, HIDDEN_LAYER_SIZE+1))
	#print np.shape(theta1)
	#print np.shape(theta2)
	m = np.shape(x_data)[0]
	J = 0
	theta1_grad = np.matrix(np.zeros(np.shape(theta1)))
	theta2_grad = np.matrix(np.zeros(np.shape(theta2)))
	bigdelta1 = 0
	bigdelta2 = 0

	for t in range(m):
		#a1 = np.append(1, x_data[t, :])
		a1 = x_data[t, :]
		z2 = theta1 * np.transpose(a1)
		a2 = sigmoid(z2)
		a2 = np.transpose(a2)
		#print np.shape(a2)
		#print a2
		a2 = np.matrix(np.append(1, a2))
		#print a2
		#print np.shape(a2)
		#print z2
		z3 = theta2 * np.transpose(a2)
		a3 = sigmoid(z3)
		a3 = np.transpose(a3)
		hTheta = a3
		y_temp = np.zeros((NUM_LABELS, 1))
		y_temp[y_data[t, 0]-1, 0] = 1
		J += np.log(hTheta) * (-y_temp) - np.log(1-hTheta) * (1 - y_temp)
		delta3 = np.transpose(a3) - y_temp
		#print 'delta3'
		#print type(delta3)
		#print delta3
		#raw_input()
		delta2 = np.multiply((np.transpose(theta2[:, 1:]) * delta3), sigmoid_gradient(z2))
		bigdelta2 = bigdelta2 + delta3 * a2
		bigdelta1 = bigdelta1 + delta2 * a1
	J = (1.0/m)*J + ((lambda_param/(2.0*m)) * ((np.sum(np.power(theta1[:, 1:], 2))) + (np.sum(np.power(theta2[:, 1:], 2)))))
	theta2_grad = (1.0/m)*bigdelta2 + (lambda_param *1.0/m) * np.append(np.zeros((np.shape(theta2)[0],1)), theta2[:, 1:], axis=1)
	theta1_grad = (1.0/m)*bigdelta1 + (lambda_param *1.0/m) * np.append(np.zeros((np.shape(theta1)[0],1)), theta1[:, 1:], axis=1)
	grad = np.concatenate((np.asarray(theta1_grad).flatten(), np.asarray(theta2_grad).flatten()))
	return J, grad

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

mat_contents = sio.loadmat('ex4data1.mat')
#print mat_contents
x_data = mat_contents['X']
constants = np.ones((np.shape(x_data)[0], 1))
x_data = np.matrix(np.append(constants, x_data, axis=1))
#print x_data
#print np.shape(x_data)
y_data = np.matrix(mat_contents['y'])
#print np.shape(y_data)
weights = sio.loadmat('ex4weights.mat')
#print weights
theta1 = weights['Theta1']
theta2 = weights['Theta2']
print np.shape(theta1)
print np.shape(theta2)
nn_params = np.concatenate((np.asarray(theta1).flatten(), np.asarray(theta2).flatten()))
lambda_param = 0
J, grad = nn_cost_function(nn_params, x_data, y_data, lambda_param)
print J
lambda_param = 1
J, grad = nn_cost_function(nn_params, x_data, y_data, lambda_param)
print J
sg = sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print sg
initial_theta1 = random_initialize_weights(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE)
initial_Theta2 = random_initialize_weights(HIDDEN_LAYER_SIZE, NUM_LABELS)
initial_nn_params = np.concatenate((np.asarray(initial_theta1).flatten(), np.asarray(initial_Theta2).flatten()))
fmin = minimize(fun=nn_cost_function, x0=initial_nn_params, args=(x_data, y_data, lambda_param), method='TNC', jac=True, options={'maxiter': 250})
print fmin
theta1 = np.matrix(np.reshape(fmin.x[:HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1)], (HIDDEN_LAYER_SIZE, (INPUT_LAYER_SIZE + 1))))  
theta2 = np.matrix(np.reshape(fmin.x[HIDDEN_LAYER_SIZE * (INPUT_LAYER_SIZE + 1):], (NUM_LABELS, (HIDDEN_LAYER_SIZE + 1))))
p = predict(theta1, theta2, x_data)
accuracy = (y_data[np.where(y_data==p)].size * 1.0 / y_data.size) * 100
print 'accuracy is ', accuracy
