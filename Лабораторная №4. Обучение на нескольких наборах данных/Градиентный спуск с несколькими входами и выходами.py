import numpy as np

def neural_network(inp, weights):
    return inp.dot(weights)

def get_error(true_prediction, prediction):
    return np.mean((true_prediction - prediction)**2)

inp = np.array([120, 30])
weights = np.array([[0.2, 0.3],
                    [0.5, 0.7]]).T
true_prediction = np.array([100, 40])
learning_rate = 1e-5  # уменьшил, чтобы шаг был стабильный

for i in range(30):
    prediction = neural_network(inp, weights)
    error = get_error(true_prediction, prediction)
    
    # градиент ошибки по весам
    delta = np.outer(inp, (prediction - true_prediction))
    
    # обновляем веса
    weights -= learning_rate * delta
    print(f"Step {i}: Prediction: {prediction}, Error: {error}")
