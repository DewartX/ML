import numpy as np

def neural_networks(inp, weight):
	return inp * weight

def get_error(true_pred, pred):
	return (true_pred - pred) ** 2

inp = 3
weight = 0.4

learning_rate = 0.01

true_pred = 0.8

print(get_error(true_pred, neural_networks(inp, weight)))

for i in range(40): #оптимальное число итераций
    pred = neural_networks(inp, weight)
    error = get_error(true_pred, pred)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" %(pred, weight, error))
    delta = (pred - true_pred) * inp
    weight -= learning_rate * delta
