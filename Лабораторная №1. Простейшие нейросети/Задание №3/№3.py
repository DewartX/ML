def neuralNetwork(inp, bias):
    weight = 0.5
    prediction = inp * weight + bias
    return prediction


inputs = [150, 160, 170, 180, 190]
bias = 10 #случайно подобранное смещение

for inp in inputs:
    output = neuralNetwork(inp, bias)
    print(f"Вход: {inp}, Выход: {output}")