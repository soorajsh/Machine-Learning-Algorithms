import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred ) / len(y_true)
    return accuracy

regressor = LogisticRegression(lr = 0.0001, n_iters = 1000)
regressor.fit(x_train, y_train)
predictions = regressor.predict(x_test)


print("LR classification accuracy: ", accuracy(y_test, predictions))