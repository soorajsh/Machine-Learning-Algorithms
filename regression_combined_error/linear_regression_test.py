import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


x,y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=1234)


from regression_combined import LinearRegression
regressor = LinearRegression( lr=0.01)
regressor.fit(x_train, y_train)
predicted = regressor.predict(x_test)

def mean_square_error(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

mse_value = mean_square_error(y_test, predicted)
print(mse_value)

# figure 

y_pred_line = regressor.predict(x)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
n1 = plt.scatter(x_train, y_train, color =cmap(0.9), s=10)
n2 = plt.scatter(x_test, y_test, color= cmap(0.5), s=10)
plt.plot(x,y_pred_line, color='black', linewidth=2, label = "Prediction")
plt.show()
