import pandas as pd
import numpy as np
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

def gradient_descent(x_data, y_data, theta, alpha, iterations, m):
	#hTheta = None
	for i in range(iterations):
		#hTheta = np.transpose(np.transpose(theta) * np.transpose(x_data))
		hTheta = x_data * theta
		theta = theta - alpha*(1.0/m)*np.transpose(np.transpose(hTheta - y_data) * x_data)

	return theta

def computeCost(x_data, y_data, theta, m):
	J = 0
	#hTheta = np.transpose(np.dot(np.transpose(theta), np.transpose(x_data)))
	#hTheta = np.transpose(np.transpose(theta) * np.transpose(x_data))
	hTheta = x_data * theta
	#print hTheta
	#print (1.0/(2*m))
	print 'np.shape of hTheta: ', np.shape(hTheta)
	print 'np.shape of y_data: ', np.shape(y_data)
	#print hTheta
	#print y_data
	#print np.transpose(y_data)
	#print hTheta - y_data
	#print np.sum((hTheta - y_data)**2)
	J = (1.0/(2*m))*np.sum(np.square(hTheta - y_data))
	return J

	#hTheta = (theta' * X')';
#%size(hTheta)
#J = (1/(2*m))*sum((hTheta - y).^2);

df = pd.read_csv('ex1data1.csv', header=0)
df.info()
df['Constant'] = 1
df.info()
#df.reindex_axis(sorted(df.columns), axis=1)
df = df.reindex_axis(['Constant', 'Population', 'Profit'], axis=1)
df.info()
x_data = np.matrix(df.drop('Profit', axis=1).values)
y_data = np.transpose(np.matrix(df['Profit'].values))
print np.shape(x_data)
print np.shape(y_data)
#print x_data
#print y_data

#Plot the data
scatter(x_data[:, 1], y_data[:, 0], marker='o', c='b')
title('Profits distribution')
xlabel('Population of City in 10,000s')
ylabel('Profit in $10,000s')
#show()

m = np.shape(y_data)[0]
print m
iterations = 1500
alpha = 0.01
theta = np.zeros((2, 1))
print theta
print computeCost(x_data, y_data, theta, m)
theta = gradient_descent(x_data, y_data, theta, alpha, iterations, m)

#Plot the results
result = x_data * theta
plot(x_data[:, 1], result)
show()

print 'theta is ', theta
predict1 = (np.matrix([1, 3.5]) * theta).item(0)
print 'For population = 35,000, we predict a profit of ', predict1*10000
predict2 = (np.matrix([1, 7]) * theta).item(0)
print 'For population = 70,000, we predict a profit of ', predict2*10000




