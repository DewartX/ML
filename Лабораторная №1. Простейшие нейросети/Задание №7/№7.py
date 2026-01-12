# Функция нейросети
def neuralNetwork(inp, weights):
    prediction = [0] * len(weights) 

    for i in range(len(weights)):
        ws = 0 
        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]
        prediction[i] = ws

    return prediction


inp = [50, 165]

# Исходные наборы весов
weights_1 = [0.2, 0.1]
weights_2 = [0.3, 0.1]

# Новый набор весов
weights_3 = [0.4, 0.2]

# Список всех весов
weights = [weights_1, weights_2, weights_3]

# Запускаем функцию
output = neuralNetwork(inp, weights)

print("Выходы для всех наборов весов:", output)
