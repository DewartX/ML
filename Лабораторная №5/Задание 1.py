import numpy as np

def neural_network(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2)) #конкретно значение ошибки или среднеквадрат. отклонения не влияет на обучение, влияет градиент
inp = np.array([
    [150, 40],
    [140, 35],
    [155, 45],
    [185, 95],
    [145, 40],
    [195, 100],
    [180, 95],
    [170, 80],
    [160, 90],
])

true_predictions = np.array([0,0,0,100,0,100,100,100,100])

weights = np.array([0.2, 0.3])
learning_rate = 0.000001 #уменьшил в 10 раз значение, ошибка сильно увеличилась
epochs = 500

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}")
    for j in range(len(inp)):
        current_inp = inp[j]
        true_prediction = true_predictions[j]
        prediction = neural_network(current_inp, weights)
        error = get_error(true_prediction, prediction)
        print("True_prediction: %.10f, Prediction: %.10f, Weights: %s" % (true_prediction, prediction, weights))
        delta = (prediction - true_prediction) * current_inp * learning_rate
        weights = weights - delta
    print("-------------------")

print("Final:")
error_total = 0
for j in range(len(inp)):
    current_inp = inp[j]
    true_prediction = true_predictions[j]
    prediction = neural_network(current_inp, weights)
    error = get_error(true_prediction, prediction)
    error_total += error
    print("True_prediction: %.10f, Prediction: %.10f, Weights: %s" % (true_prediction, prediction, weights))
print("Total Error: %.10f" % error_total)

print(neural_network(np.array([150,45]), weights))
print(neural_network(np.array([170,85]), weights))