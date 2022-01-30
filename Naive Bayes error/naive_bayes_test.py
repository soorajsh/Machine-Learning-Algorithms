import numpy as np
from scipy.sparse import data

from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from naive_bayes import NaiveBayes

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

x,y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=123)

nb = NaiveBayes()
nb.fit(x_train, y_train)
predictions = nb.predict(x_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))