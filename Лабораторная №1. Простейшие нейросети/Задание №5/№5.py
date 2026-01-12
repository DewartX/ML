# Исходный код
# def neuralNetwork(inp, weights):
#     prediction = [0,0]
#     for i in range(len(weights)):
#         prediction[i] = inp * weights[i]
#     return prediction
#
# print(neuralNetwork(4, [0.2, 0.5]))


def neuralNetwork(inp, weights):
    prediction = [0, 0]
    for i in range(len(weights)):
        prediction[i] = inp * weights[i]
    return prediction

# Входное значение
inp = 4

# Подобранные веса
weights = [0.125, 0.5]  # при inp = 4, значение больше 0.5 получается при весе 0.125

# Вывод результатов
output = neuralNetwork(inp, weights)
print(f"Выход: {output}")
