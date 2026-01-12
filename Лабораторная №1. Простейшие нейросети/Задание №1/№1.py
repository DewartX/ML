def neuralNetwork(inp, weights):
    # weights — двумерный массив,
    # каждая строка соответствует одному выходному нейрону

    prediction = [0, 0] #всего два выхода

    for i in range(len(weights)):
        ws = 0  # взвешенная сумма для каждого выходного нейрона
        for j in range(len(inp)):
            ws += inp[j] * weights[i][j]
        prediction[i] = ws

    return prediction


# Версия 1 — исходные данные

inp_1 = [10, 20] 

weights_1 = [
    [0.1, 0.2],   # веса для первого выходного нейрона
    [0.3, 0.4]    # веса для второго выходного нейрона
]

output_1 = neuralNetwork(inp_1, weights_1)

print("Версия 1 (исходные данные)")
print("Входные данные:", inp_1)
print("Веса:", weights_1)
print("Выход:", output_1)
print()



# Версия 2 — изменённые данные
# Увеличил входные значения и веса нейросети в 2 раза

inp_2 = [20, 40]

weights_2 = [
    [0.2, 0.4], 
    [0.6, 0.8]
]

output_2 = neuralNetwork(inp_2, weights_2)

print("Версия 2 (изменённые данные)")
print("Входные данные:", inp_2)
print("Веса:", weights_2)
print("Выход:", output_2)
