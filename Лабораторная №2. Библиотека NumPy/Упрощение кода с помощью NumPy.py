import numpy as np

# Вход
inp = np.array([23, 45])

# Генерация случайных весов
weights_h1 = np.random.rand(2, 2)  # первый скрытый слой 2 нейрона x 2 входа
weights_h2 = np.random.rand(2, 2)  # второй скрытый слой 2 нейрона x 2 входа
weights_out = np.random.rand(2, 1) # выход 1 нейрон x 2 входа из второго скрытого слоя

weights = [weights_h1, weights_h2, weights_out]

def neuralNetwork(inp, weights):
    h1 = inp.dot(weights[0])       # первый скрытый слой
    h2 = h1.dot(weights[1])        # второй скрытый слой
    out = h2.dot(weights[2])       # выход
    return out

print("Выход нейросети:", neuralNetwork(inp, weights))
