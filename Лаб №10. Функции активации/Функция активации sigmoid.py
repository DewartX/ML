import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) #с функцией активации relu, ошибка уменьшилась примерно раз в 100, но не знаю насколько это будет плохо с точки зрения переобучения модели

def sigmoid_deriv(x):
    return x*(1-x)

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_size = x.shape[1]
hidden_size = 20 #c увеличением количества нейронов в скрытом слое с 4 до 20 ошибка значительно уменьшилась, но я не понял какая корреляция между количеством, так как с количеством 50 ошибка сильно увеличивается
output_size = y.shape[1]

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 100000

for epoch in range(epochs):
    layer_hid = sigmoid(np.dot(x, weight_hid))
    layer_out = sigmoid(np.dot(layer_hid, weight_out))
    layer_out_delta = (layer_out - y) * sigmoid_deriv(layer_out)
    layer_hid_delta = layer_out_delta.dot(weight_out.T) * sigmoid_deriv(layer_hid)
    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hid_delta)
    error = (layer_out - y) ** 2
    if epoch % 1000 == 0:
    	error = np.mean(error)
    	print(f"Epoch: {epoch}, Error: {error}")

# тест
new_input = np.array([[0,1],[1,1],[1,0],[0,0]])
layer_hid = sigmoid(np.dot(new_input, weight_hid))
layer_out = sigmoid(np.dot(layer_hid, weight_out))
print(layer_out)
