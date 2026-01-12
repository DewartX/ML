def neuralNetwork(inp, weights):
    prediction = [0, 0]
    for i in range(len(weights)):
        prediction[i] = inp * weights[i]
    return prediction


inp = 4
weights = [0.0, 0.0]      # начинаем с нулевых весов
step = 0.001              # маленький шаг
done = [False, False]     # достиг ли выход > 0.5


while not all(done):
    output = neuralNetwork(inp, weights)

    for i in range(len(weights)):
        if output[i] <= 0.5:
            weights[i] += step
        else:
            done[i] = True

print("Найденные веса:", weights)
print("Выход:", neuralNetwork(inp, weights))
