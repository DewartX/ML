def neuralNetwork(inp, weight):
    return inp * weight


inputs = [150, 160, 170, 180, 190]
weight = 0.5

for inp in inputs:
    output = neuralNetwork(inp, weight)
    print(f"Вход: {inp}, Выход: {output}")