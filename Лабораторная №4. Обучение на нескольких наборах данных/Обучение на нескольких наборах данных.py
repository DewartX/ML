import numpy as np

def neural_network(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return np.sqrt(np.mean((true_prediction - prediction) ** 2)) #смена ошибки на среднеквадратичное отклонение никак не повлияло на ошибку, так как в изменении весов она не учавствует

inp = np.array([
    [150, 40],
    [170, 80],
    [160, 90]
])

true_predictions = np.array([50, 120, 140])

weights = np.array([0.2, 0.3])
learning_rate = 0.00005
epochs = 100 #уменьшил кол-во эпох до 100, ошибка конкретно в этой ситуации увеличилась

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
