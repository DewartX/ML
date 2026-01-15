import numpy as np

def tanh(x):
    return np.tanh(x) #relu быстрее всего, но не знаю насколько точна без dropout

# Производная tanh
def tanh_deriv(x):
    return 1 - x**2

# Входные данные XOR
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

input_size = x.shape[1]
hidden_size = 4
output_size = y.shape[1]

np.random.seed(1)
weight_hid = np.random.uniform(size=(input_size, hidden_size))
weight_out = np.random.uniform(size=(hidden_size, output_size))

learning_rate = 0.1
epochs = 10000

for epoch in range(epochs):
    layer_hid = tanh(np.dot(x, weight_hid))
    layer_out = tanh(np.dot(layer_hid, weight_out))
    
    layer_out_delta = (layer_out - y) * tanh_deriv(layer_out)
    layer_hid_delta = layer_out_delta.dot(weight_out.T) * tanh_deriv(layer_hid)

    weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
    weight_hid -= learning_rate * x.T.dot(layer_hid_delta)
    
    if epoch % 1000 == 0:
        error = np.mean((layer_out - y)**2)
        print(f"Epoch: {epoch}, Error: {error:.5f}")

# Тест
new_input = np.array([[0,0],[0,1],[1,0],[1,1]])
layer_hid = tanh(np.dot(new_input, weight_hid))
layer_out = tanh(np.dot(layer_hid, weight_out))
print("Результаты после обучения:")
print(layer_out)
