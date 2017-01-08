import pandas as pd
import numpy as np
from pylab import scatter, show, legend, xlabel, ylabel
import math
import scipy.optimize as opt

def sigmoid(z):
	return 1.0/(1 + np.power(math.e, -z))

def grad(theta, x_data, y_data):
	#print np.shape(x_data)
	#print np.shape(theta)
	m = np.shape(y_data)[0]
	n = np.shape(x_data)[1]
	hTheta = sigmoid(x_data * theta.reshape((n,1)))
	grad = (1.0/m)*np.transpose(np.transpose(hTheta - y_data) * x_data)
	#print grad
	#print np.squeeze(np.asarray(grad))
	return np.squeeze(np.asarray(grad))

def computeCost(theta, x_data, y_data):
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
	return J.item(0)

def predict(theta, x_data):
	m = np.shape(x_data)[0]
	n = np.shape(x_data)[1]
	p = np.zeros((m, 1))
	hTheta = sigmoid(x_data * theta.reshape((n,1)))
	p = np.round(hTheta)
	#print np.shape(hTheta)
	return p

df = pd.read_csv('ex2data1.csv', header=0)
df.info()
df['Constant'] = 1
df.info()
#df.reindex_axis(sorted(df.columns), axis=1)
df = df.reindex_axis(['Constant', 'Marks1', 'Marks2', 'Result'], axis=1)
df.info()
x_data = np.matrix(df.drop('Result', axis=1).values)
y_data = np.transpose(np.matrix(df['Result'].values))
print np.shape(x_data)
print np.shape(y_data)
#print x_data
#print y_data
#m = np.shape(y_data)[0]
#print m
n = np.shape(x_data)[1]
pos = np.where(y_data == 1)
neg = np.where(y_data == 0)
scatter(x_data[pos, 1], x_data[pos, 2], marker='o', c='b')
scatter(x_data[neg, 1], x_data[neg, 2], marker='x', c='r')
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend(['Not Admitted', 'Admitted'])
#show()
#raw_input()
theta = np.zeros(n)
print theta
print computeCost(theta, x_data, y_data)
print grad(theta, x_data, y_data)
#raw_input()
#print np.shape(theta)
theta = opt.fmin_bfgs(computeCost, theta, fprime=grad, args=(x_data, y_data))
#theta = gradient_descent(x_data, y_data, theta, alpha, iterations, m)
print 'theta is ', theta
prob = np.dot(np.array([1, 45, 85]), theta)
prob = sigmoid(prob)
print prob
p = predict(theta, x_data)
accuracy = (y_data[np.where(y_data==p)].size * 1.0 / y_data.size) * 100
print 'accuracy is ', accuracy
#predict1 = (np.matrix([1, 3.5]) * theta).item(0)
#print 'For population = 35,000, we predict a profit of ', predict1*10000
#predict2 = (np.matrix([1, 7]) * theta).item(0)
#print 'For population = 70,000, we predict a profit of ', predict2*10000




