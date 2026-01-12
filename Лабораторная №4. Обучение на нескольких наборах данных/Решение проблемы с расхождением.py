import numpy as np

def neural_networks(inp, weight):
    return inp * weight

def get_error(true_pred, pred):
    return (true_pred - pred) ** 2

inp = 30
weight = 0.2
true_pred = 70
learning_rate = 0.001

for i in range(10):
    pred = neural_networks(inp, weight)
    error = get_error(true_pred, pred)
    print("Prediction: %.10f, Weight: %.5f, Error: %.20f" % (pred, weight, error))
    
    # вычисляем delta заново на каждом шаге
    delta = (pred - true_pred) * inp
    weight -= learning_rate * delta

print("Final weight:", weight)
print("Final error:", get_error(true_pred, neural_networks(inp, weight)))
