# Linear Model of Support Vector Machines

import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param = 0.01, n_iters =1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w None
        self.b = None

    def fit(self,x,y):
        y_ = np.where(y <=0, -1, 1)
        

    def predict(self,x):
        linear_output = np.dot(x, self.w)- self.b
        return np.sign(linear_output)

