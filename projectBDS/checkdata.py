import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
from tkinter.ttk import *

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

win = Tk()

win.title("Home price prediction")
win.geometry("1500x2630")
win.iconbitmap("logo.ico")


dataset = pd.read_csv("123.csv").values

X = dataset[:, 0:4].reshape(-1, 4)

X = np.hstack((np.ones((X.shape[0], 1)), X))

y = dataset[:, dataset.shape[1]-1]

n, d = X.shape

theta = np.array([0.1]*d)
print(theta)

iters = 1000
learning_rate = 0.00001


#-------------------------------
datatest = pd.read_csv("dataBDStest.csv").values

Ytest = datatest[:, datatest.shape[1]-1]

Xtest = datatest[:, 0:4].reshape(-1, 4)

Xtest = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))
# #-------------------------------


def cost_function(theta, X, y):
    y1 = theta*X
    y1 = np.sum(y1, axis=1)
    return (1/(2*n))*sum(y-y1)**2

def train(theta, X, y, learning_rate, iters):
    cost_history = []
    for i in range(iters):
        for c in range(d):
            # y1 = theta*X
            y1 = theta * X
            y1 = np.sum(y1, axis=1)
            theta[c] = theta[c]-learning_rate*1/n*sum((y1-y)*X[:, c])
        cost = cost_function(theta, X, y)
        cost_history.append(cost)
    return theta, cost_history

def prediction():
    theta1, cost_history = train(theta, X, y, learning_rate, iters)
    t3.delete(0, 'end')
    num = float(t.get())
    num0 = float(t0.get())
    num1 = float(t1.get())
    num2 = float(t2.get())

    X1 = np.array([[1, num, num0, num1, num2]])
    y1 = theta1 * X1
    plt.figure()
    result = np.sum(y1, axis=1)
    t3.insert(END, str(result))

def clean():
    t.delete(0, 'end')
    t0.delete(0, 'end')
    t1.delete(0, 'end')
    t2.delete(0, 'end')
    t3.delete(0, 'end')

lbl = Label(win, text='Acreage(m^2):')
lbl0 = Label(win, text='Street:')
lbl1 = Label(win, text='Floors:')
lbl2 = Label(win, text='Rooms:')
lbl3 = Label(win, text='Prediction(B):')

t = Entry()
t0 = Entry()
t1 = Entry()
t2 = Entry()
t3 = Entry()

btn1 = Button(win, text='Prediction')
btn2 = Button(win, text='Clean')

lbl.place(x=10, y=20)
t.place(x=100, y=20)
lbl0.place(x=10, y=60)
t0.place(x=100, y=60)
lbl1.place(x=10, y=100)
t1.place(x=100, y=100)
lbl2.place(x=10, y=140)
t2.place(x=100, y=140)

theta1, cost_history = train(theta, X, y, learning_rate, iters)

nptheta = np.array(theta1)
npXtest = np.array(Xtest)

# print(nptheta)
# print(npXtest)

YtestTheta = np.dot(npXtest, nptheta)
# print(YtestTheta)

St = ((Ytest - Ytest.mean())**2).sum()
Sr = ((Ytest - YtestTheta)**2).sum()
R2 = 1 - (Sr/St)
print(R2)

figure = Figure(figsize=(5, 4), dpi=70)
plot = figure.add_subplot(1, 1, 1)
plot.plot(YtestTheta, Ytest, color="blue", marker="o", linestyle="")
plot.set_xlabel("Y du doan")
plot.set_ylabel("Y thuc te")
canvas = FigureCanvasTkAgg(figure, win)
canvas.get_tk_widget().place(x=250, y=10)
plot.set_title('Interest Rate Vs. Stock Index Price')
x0 = np.linspace(min(YtestTheta), max(YtestTheta))
plot.plot(x0, x0)

figure2 = Figure(figsize=(5, 4), dpi=70)
plot1 = figure2.add_subplot(1, 1, 1)
plot1.plot(cost_history)
canvas2 = FigureCanvasTkAgg(figure2, win)
canvas2.get_tk_widget().place(x=250, y=350)
plot1.set_title('Interest Rate Vs. Stock Index Price')

b1 = Button(win, text='Prediction', command = prediction)
b2 = Button(win, text='Clean', command = clean)

b2.bind('<Button-1>')
b1.place(x=40, y=190)
b2.place(x=150, y=190)

lbl3.place(x=10, y=230)
t3.place(x=100, y=230)

win.geometry("620x750+10+10")
win.mainloop()