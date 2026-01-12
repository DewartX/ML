def neuralNetwork(inps, weights):
    prediction = 0
    between = [] # список промежуточных значений

    for i in range(len(weights)):
        prom = inps[i] * weights[i]
        between.append(prom)
        prediction += prom

    return prediction, between  # возвращаем оба значения


prediction_1, between_1 = neuralNetwork([150, 40], [0.3, 0.4])
prediction_2, between_2 = neuralNetwork([80, 60], [0.2, 0.4])

print(f"Выход: {prediction_1}, Промежуточные значения: {between_1}")
print(f"Выход: {prediction_2}, Промежуточные значения: {between_2}")
