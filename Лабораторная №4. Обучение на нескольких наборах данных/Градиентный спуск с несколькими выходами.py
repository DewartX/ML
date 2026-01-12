import numpy as np

def neural_networks(inp, weights):
    return inp*weights

def get_error(true_prediction, prediction):
    return (true_prediction - prediction) ** 2

inp = 200
weights = np.array([0.2,0.3])
true_prediction = ([100,230])
learning_rate = 0.00001 #бОльшие значения вызывают большие расхождения 

for i in range(30):
    prediction = neural_networks(inp, weights)
    error = get_error(true_prediction, prediction)
    print("Prediction: %s, Weights: %s, Error: %s" % (prediction, weights, error))
    delta = (prediction - true_prediction) * inp
    weights = weights - delta * learning_rate
