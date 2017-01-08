import pandas as pd
import numpy as np

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
	#print 'np.shape of hTheta: ', np.shape(hTheta)
	#print 'np.shape of y_data: ', np.shape(y_data)
	#print hTheta
	#print y_data
	#print np.transpose(y_data)
	#print hTheta - y_data
	#print np.sum((hTheta - y_data)**2)
	J = (1.0/(2*m))*np.sum(np.square(hTheta - y_data))
	return J

def normal_equation(x_data, y_data):
	theta = np.zeros((np.shape(x_data)[1], 1))
	#print theta
	theta = np.linalg.pinv(np.transpose(x_data) * x_data) * np.transpose(x_data) * y_data
	return theta
	#hTheta = (theta' * X')';
#%size(hTheta)
#J = (1/(2*m))*sum((hTheta - y).^2);

df = pd.read_csv('ex1data2.csv', header=0)
#df.info()
mean = df.mean()
std = df.std()
#print df.mean()
#print df.std()
#print mean.Area
#print std.Bedrooms
df['Constant'] = 1

''' Normalize the Area and Bedrooms columns '''
df['norm_area'] = df['Area']
df.norm_area = (df.norm_area - df.norm_area.mean()) / df.norm_area.std()
df['norm_bedrooms'] = df['Bedrooms']
df.norm_bedrooms = (df.norm_bedrooms - df.norm_bedrooms.mean()) / df.norm_bedrooms.std()
#print df.head(10)
#raw_input()
#df.info()
#df.reindex_axis(sorted(df.columns), axis=1)
#df = df.reindex_axis(['Constant', 'Area', 'Bedrooms', 'Cost'], axis=1)
#df.info()
x_data = np.matrix(df.drop(['Area', 'Bedrooms', 'Cost'], axis=1).values)
y_data = np.transpose(np.matrix(df['Cost'].values))
#print x_data
#print np.shape(x_data)
#print np.shape(y_data)
#print x_data
#print y_data
m = np.shape(y_data)[0]
#print m

''' prediction using gradient descent '''
iterations = 400
alpha = 0.01
theta = np.zeros((3, 1))
#print theta
#print computeCost(x_data, y_data, theta, m)
theta = gradient_descent(x_data, y_data, theta, alpha, iterations, m)
print 'theta is ', theta
#predict1 = (np.matrix([1, 3.5]) * theta).item(0)
#print 'For population = 35,000, we predict a profit of ', predict1*10000
#predict2 = (np.matrix([1, 7]) * theta).item(0)
#print 'For population = 70,000, we predict a profit of ', predict2*10000
price = 0
x_predict = np.matrix([1, 1650, 3], dtype = np.float)
#print x_predict
#print mean.Area
#print std.Area
#print (x_predict[0, 1] - mean.Area) / std.Area
x_predict[0, 1] = (x_predict[0, 1]*1.0 - mean.Area) / std.Area
x_predict[0, 2] = (x_predict[0, 2]*1.0 - mean.Bedrooms) / std.Bedrooms
#print x_predict
price = (x_predict * theta).item(0)
print 'price for area = 1650 and 3 bedrooms using gradient descent is $' + str(price) + '\n'

''' prediction using normal equation '''
df = df.reindex_axis(['Constant', 'Area', 'Bedrooms', 'Cost', 'norm_area', 'norm_bedrooms'], axis=1)
#df.info()
orig_xdata = np.matrix(df.drop(['Cost', 'norm_area', 'norm_bedrooms'], axis=1).values)
theta = normal_equation(orig_xdata, y_data)
print 'theta is ', theta

'''
price = 0; % You should change this
xpredict = [1650 3];
xpredict(1, :) = (xpredict(1, :) - mu) ./ sigma;
xpredict = [1, xpredict];
price = theta' * xpredict';
'''
price = 0
x_predict = np.matrix([1, 1650, 3], dtype = np.float)
price = (x_predict * theta).item(0)
print 'price for area = 1650 and 3 bedrooms using normal equation is $' + str(price)
