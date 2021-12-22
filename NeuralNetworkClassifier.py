import numpy as np #No using automatic differentiation allowed from here!
import pandas as pd 
from math import log, exp

class NeuralNetworkClassifier:
    def relu(x):
        return np.array([max(0, x[i]) for i in range(len(x))])

    def softmax(self, x):
        exp_sum = sum(exp(x[i]) for i in range(len(x)))
        return np.array([exp(x[i]) / exp_sum for i in range(len(x))])
    
    def linear(self, x, l):
        m, c = self.lin_coeffs[l]
        return np.array([m * x[i] + c for i in range(len(x))])
    
    def derivative_relu(x):
        return np.array([int(x[i] > 0) for i in range(len(x))])

    def derivative_softmax(self, x):
        s = [self.softmax(x[i]) for i in range(len(x))] 
        return [[s[i] * (int(i == j) - s[j]) for j in range(len(x))] for i in range(len(x))]

    def derivative_linear(self, l):
        return self.lin_coeffs[l][0]
    
    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)
        
        self.lin_coeffs = {}
        for i, layer in enumerate(layers):
            prev, next, activation = layer
            
            if activation == 'linear':
                self.lin_coeffs[i] = (np.random.rand(), np.random.rand())

        self.wts = [] + [np.random.randn(next, prev + 1) for prev, next, activation in layers]
        self.act = []
        self.z = []

    def forward_propogation(self, x):
        x.append(self.BIAS)
        self.act = [x] + self.act
        
        for l in range(1, self.L + 1):
            self.z[l] = self.wts[l] @ self.act[l - 1]
            activation = self.layers[l][2]
            self.act[l] = self.ACTIVATION[activation](self.z[l], l) if activation == 'linear' else self.ACTIVATION[activation](self.z[l])  

        return self.act[-1]
    
    def backward_propogation(self, y):
        for l in range(1, self.L + 1):
            for i in range(len(self.wts[l])):
                for j in range(len(self.wts[l][i])):
                    if i == y:
                        activation = self.layers[l - 1][2]
                        derivative = self.DERIVATIVE[activation](l) if activation == 'linear' else self.DERIVATIVE[activation](self.z[l][y]) 
                        scalers = self.alpha * self.act[l - 1][j] / float(self.act[l][y])
                        self.wts[i][j] -= scalers * derivative 

    def fit_once(self, X, Y, alpha):
        self.alpha = alpha
        
        for x, y in list(zip(X, Y)):
            self.forward_propogation(x)
            self.backward_propogation(y)
    
    def predict(self, x):
        return self.forward_propogation(x)
    
    def categorical_cross_entropy_loss(self, y, yhat):
        return -log(yhat[y])
    
    BIAS = 1
    ACTIVATION = {'relu': relu, 'softmax': softmax, 'linear': linear}
    DERIVATIVE = {'relu': derivative_relu, 'softmax': derivative_softmax, 
                  'linear': derivative_linear}
    
NUM_ROWS = 10000
NUM_COLUMNS_X = 200 
NUM_CLASSES = 5
X = np.random.uniform(size=(NUM_ROWS,NUM_COLUMNS_X))
y = np.random.randint(size=(NUM_ROWS,),low=0,high=NUM_CLASSES)

model = NeuralNetworkClassifier([(NUM_COLUMNS_X, "relu"), (200, "relu"), (NUM_CLASSES, "softmax")])

losses = [] 
NUM_ITERS = 100
for _ in range(NUM_ITERS):
    yhat = model.predict(X, y)
    loss = model.categorical_cross_entropy_loss(y, yhat)
    losses.append(loss)
    model.fit_once(X, y)