def neuralNetwork(inp, weights):
    prediction_h = [0] * len(weights)  # предсказания скрытого слоя
    for i in range(len(weights)):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]
        prediction_h[i] = ws
    return prediction_h

# Входные данные
inp = [23, 45]

# Начальные веса скрытого слоя
weight_h_1 = [0.4, 0.1]
weight_h_2 = [0.3, 0.2]

weights_h = [weight_h_1, weight_h_2]

step = 0.01 
while True:
    prediction_h = neuralNetwork(inp, weights_h)
    
    # проверяем, все ли нейроны > 5
    if all(p > 5 for p in prediction_h):
        break

    # увеличиваем веса только тех нейронов, где < 5
    for i in range(len(weights_h)):
        if prediction_h[i] < 5:
            for j in range(len(weights_h[i])):
                weights_h[i][j] += step

print("Получившиеся веса скрытого слоя:", weights_h)
print("Предсказания скрытого слоя:", neuralNetwork(inp, weights_h))
