import numpy as np
import pandas as pd 
from math import log
from ActivationFunctions import *

class NeuralNetworkClassifier:
    BIAS = 1
    ACTIVATION = {'relu': relu, 'softmax': softmax, 'linear': linear}
    DERIVATIVE = {'relu': derivative_relu, 'softmax': derivative_softmax, 
                  'linear': derivative_linear}
    CORRECTION = 1e-6
    
    def __init__(self, layers):
        self.layers = layers
        self.L = len(layers)
        
        self.lin_coeffs = {}
        prev_next = 0
        for l, layer in enumerate(layers):
            prev, next, activation = layer
            
            if l > 0 and prev_next != prev:
                raise Exception("The no. of layers in this layer is inconsistent."
                                + "\nThe last layer mentioned no. of layers in " 
                                + str(l) + "th layer is " + str(prev_next) 
                                + ".\nBut this layer mentions the no. of layers to be " 
                                + str(prev) + ".")
            
            if activation == 'linear':
                self.lin_coeffs[l] = (np.random.rand(), np.random.rand())
                
            prev_next = next            

        self.wts = [np.random.randn(next, prev + 1) for prev, next, activation in layers[:-1]]
        self.wts.append(np.random.randn(layers[-1][1], layers[-1][0] + 1))
        self.wts.insert(0, [])
        
        self.act = [[] for i in range(self.L + 1)]
        self.z = [[] for i in range(self.L + 1)]

    def _forward_propogation(self, x):
        x = np.append(x, self.BIAS)
        self.act[0] = x
        
        for l in range(1, self.L + 1):
            self.z[l] = self.wts[l] @ self.act[l - 1]
            activation = self.layers[l - 1][2]
            
            if activation == 'linear':
                m, c = self.lin_coeffs[l]
                self.act[l] = np.append(self.ACTIVATION[activation](self.z[l], m, c), 1)
            else:
                self.act[l] = np.append(self.ACTIVATION[activation](self.z[l]), 1)  

        return self.act[self.L]
    
    def _backward_propogation(self, y):
        for l in range(1, self.L):
            for i in range(len(self.wts[l])):
                for j in range(len(self.wts[l][0])):
                    if i == y:
                        activation = self.layers[l - 1][2]
                        
                        if activation == 'linear':
                            m = self.lin_coeffs[l][0]
                            derivative = self.DERIVATIVE[activation](m)
                        else:
                            derivative = self.DERIVATIVE[activation](self.z[l][y])
                             
                        # print("ZERO ERROR", float(self.act[l][y]))
                        scalers = self.alpha * self.act[l - 1][j] / float(self.act[l][y] + self.CORRECTION)
                        
                        self.wts[l][i][j] -= scalers * derivative 

    def fit_once(self, X, Y, alpha):
        n_cols = X.shape[1]
        
        if n_cols != self.layers[0][0]:
            raise Exception("The no. of neurons in the first layer should be the same as the no. of columns in X " 
                            + "\nNo. of columns in X " + str(n_cols) 
                            + "\nNo. of neurons in first layer " + str(self.layers[0][0]))  
            
        if max(Y) + 1 != self.layers[-1][1]:
            raise Exception("The no. of neurons in the last layer should be the same as the no. of classes in Y"
                            + "\nNo. of classes in Y " + str(max(Y) + 1)
                            + "\nNo. of neurons in the last layer " + str(self.layers[-1][1]))      
        self.alpha = alpha
        
        for x, y in list(zip(X, Y)):
            self._forward_propogation(x)
            self._backward_propogation(y)
    
    def predict(self, x):
        n_cols = X.shape[1]
        
        if n_cols != self.layers[0][0]:
            raise Exception("The no. of neurons in the first layer should be the same as the no. of columns in X " 
                            + "\nNo. of columns in X " + str(n_cols) 
                            + "\nNo. of neurons in first layer " + str(self.layers[0][0]))  
            
        return [self._forward_propogation(x_vec) for x_vec in x]
    
    def categorical_cross_entropy_loss(self, y, yhat):
        if max(y) + 1 != self.layers[-1][1]:
            raise Exception("The no. of neurons in the last layer should be the same as the no. of classes in Y"
                            + "\nNo. of classes in Y " + str(max(Y) + 1)
                            + "\nNo. of neurons in the last layer " + str(self.layers[-1][1])) 
        error = 0
        for i, y_label in enumerate(y):
            error += -log(yhat[i][y_label] + self.CORRECTION)
        return error
    
    
if __name__ == '__main__':
    NUM_ROWS = 1000
    NUM_COLUMNS_X = 200 
    NUM_CLASSES = 5
    X = np.random.uniform(size=(NUM_ROWS,NUM_COLUMNS_X))
    y = np.random.randint(size=(NUM_ROWS,),low=0,high=NUM_CLASSES)
    model = NeuralNetworkClassifier([(NUM_COLUMNS_X, 200, "relu"), (200, 2, "relu"), (2, NUM_CLASSES, "softmax")])

    losses = [] 
    NUM_ITERS = 4
    for _ in range(NUM_ITERS):
        yhat = model.predict(X)
        loss = model.categorical_cross_entropy_loss(y, yhat)
        print("Curr loss:", loss)
        losses.append(loss)
        print("List of losses:", losses)
        model.fit_once(X, y, 0.01)
        print("Iter no:", _)