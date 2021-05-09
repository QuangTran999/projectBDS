import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv("dataBDS.csv").values

X = dataset[:, 0:4].reshape(-1, 4) #reshape(số hàng, số cột)

X = np.hstack((np.ones((X.shape[0], 1)), X))

y = dataset[:, dataset.shape[1]-1]

len = int(len(y))

n, d = X.shape
print(n)

theta = np.array([0.1]*d)

def cost_function(theta, X, y):
    y1 = theta*X
    y1 = np.sum(y1, axis=1)
    return (1/(2*n))*sum(y-y1)**2

def train(theta, X, y, learning_rate, iters):
    cost_history = []
    for i in range(iters):
        for c in range(d):
            y1 = theta*X
            y1 = np.sum(y1, axis=1)#sum theo hang
            # print(0.5*X[:, c])
            # print(learning_rate*1/n*sum((y1-y)), "----", sum(y1-y), "-----", X[:, c])
            theta[c] = theta[c]-learning_rate*1/n*sum((y1-y)*X[:, c])#tính weight của từng cột
            # print(theta[c])
        # print("----------")
        cost = cost_function(theta, X, y)
        cost_history.append(cost)
    return theta, cost_history

def error_function():
    e=0
    iters = 100
    learning_rate = 0.0001

    theta1, cost_history = train(theta, X, y, learning_rate, iters)

    y1 = theta1*X
    y_pre = np.sum(y1, axis=1)
    print(y_pre)
    for row in range(len):
        # print(E,"---",N)
        e = e + 1/2*(y_pre[row]/y[row])**2
    E = e/len
    return E

print(error_function())




iters = 100
learning_rate = 0.00001

theta, cost_history = train(theta, X, y, learning_rate, iters)
print(cost_history)
# X1 = np.array([[1,134,6.4,3]])
y1 = theta*X
y1 = np.sum(y1, axis=1)
plt.figure()
# print(y1)
plt.scatter(x=list(range(0, n)), y=y, color='red')
# plt.scatter(x=list(range(0, n)), y=y1, color='black')
plt.show()
plt.figure()
plt.scatter(x=list(range(0, iters)), y=cost_history, color='blue')
plt.show()

