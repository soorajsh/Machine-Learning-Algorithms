import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from perceptron import Perceptron

def accuracy( y_true, y_pred):
    accuracy = np.sum(y_true == y_pred ) / len(y_true)
    return accuracy

x,y = datasets.make_blobs(n_samples=150, n_features=2, centers=2, cluster_std=1.05, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state= 123)

p = Perceptron(learning_rate=0.01, n_iters= 1000)
p.fit(x_train, y_train)
predictions = p.predict(x_test)

print('Perceptron classification accuracy ', accuracy(y_test, predictions))

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# plt.scatter(x_train[:,0], x_train[:,1], marker='O', c = y_train)

# x0_1 = 