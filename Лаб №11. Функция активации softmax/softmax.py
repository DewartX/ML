import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return (x > 0) * x


def reluderiv(x):
    return (x > 0)


def sigmoid_deriv(x):
    return x*(1-x)


def softmax(x):
    exp = np.exp(x)
    return exp/np.sum(exp, axis=1, keepdims=True)


x = np.array([
    [0,0,0,0], # 0
    [0,0,0,1], # 1
    [0,0,1,0], # 2
    [0,0,1,1], # 3
    [0,1,0,0], # 4
    [0,1,0,1], # 5
    [0,1,1,0], # 6
    [0,1,1,1], # 7
    [1,0,0,0], # 8
    [1,0,0,1], # 9
    [1,0,1,0], # 10
    [1,0,1,1], # 11
    [1,1,0,0], # 12
    [1,1,0,1], # 13
    [1,1,1,0], # 14
    [1,1,1,1]  # 15
])

y = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])



input_size = len(x[0])
hidden_size = 20
output_size = len(y[0])



np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 20000

for epoch in range(epochs):
    layer_hid = relu(np.dot(x, weight_hid))
    layer_out = softmax(np.dot(layer_hid, weight_out))
    error = (layer_out - y) ** 2

    layer_out_delta = (layer_out - y)/ len(layer_out)
    layer_hidden_delta = layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid)

    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hidden_delta)

    if epoch % 2000 == 0:
        error = np.mean(error)
        print(f"Epoch: {epoch}, Error: {error}")


def predict(inp):
    layer_hid = relu(inp.dot(weight_hid))
    layer_out = softmax(layer_hid.dot(weight_out))
    return np.argmax(layer_out)

for inp in x:
    print("=" * 50)
    print(f"Предсказанная цифра по двоичному коду{inp}:", predict(np.array([inp])))
