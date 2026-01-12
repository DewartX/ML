def neuralNetwork(inp, weights):
    prediction = [0, 0]
    for i in range(len(weights)):
        ws = 0
        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]
        prediction[i] = ws
    return prediction

inp = [50, 165]

# Начальные веса
weights_1 = [0.2, 0.1]
weights_2 = [0.3, 0.1] 
weights = [weights_1, weights_2]

# Метод проб и ошибок с циклом
while True:
    pred = neuralNetwork(inp, weights)
    diff = pred[0] - pred[1]  # разница между выходами
    if abs(diff) < 0.01:      # условие выхода
        break
    # корректируем веса второго нейрона пропорционально разнице
    for j in range(len(weights_2)):
        weights_2[j] += diff * 0.0001

print("Получившиеся веса второго нейрона:", weights_2)
print("Предсказания:", neuralNetwork(inp, weights))
