#A two-layer neural network for regression
import numpy as np
import numpy.random as npr
import sklearn.metrics
import pylab
from autograd import grad

#Generate DataSet
examples=1000
features=100
D= (npr.randn(examples,features),npr.randn(examples))

#Build the network
layer1_units = 10
layer2_units = 1
w1 = npr.rand(features,layer1_units)
b1 = npr.rand(layer1_units)
w2 = npr.rand(layer1_units,layer2_units)
b2=0.0
theta=(w1,b1,w2,b2)

#Define the loss function
def squared_loss(y,y_hat):
    return np.dot(-((y*np.log(y_hat))+ ((1-y)*np.log(1-y_hat))))

#Wrapper around the network
def neural_network(x,theta):
    w1,b1,w2,b2 = theta
    return np.tanh(np.dot((np.tanh(np.dot(x,w1)+b1)),w2) + b2)

#Wrapper around the objective function
def objective(theta,idx):
    return squared_loss(D[1][idx],neural_network(D[0][idx],theta))

#Update
def update_theta(theta,delta,alpha):
    w1,b1,w2,b2 = theta
    w1_d,b1_d,w2_d,b2_d = delta
    w1_n = w1 - alpha*w1_d
    b1_n = b1 - alpha*b1_d
    w2_n = w2 - alpha*w2_d
    b2_n = b2 - alpha*b2_d
    new_theta = (w1_n,b1_n,w2_n,b2_n)
    return new_theta

grad_objective = grad(objective)

#Train the neural network
epochs=10
rmse=[]
for i in xrange(0,epochs):
    for j in xrange(0,examples):
        delta = grad_objective(theta,j)
        theta = update_theta(theta,delta,0.01)
rmse.append()
pylad.plot(rmse)
pylab.show()
