import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return x > 0

inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])

true_prediction = np.array([[10, 20, 15, 20]]).T

layer_in = inp.shape[1]
layer_hid1 = 10 #добавил пару нейронов в первый скрытый слой, ошибка уменьшилась
layer_hid2 = 4
layer_out = 1

weights_hid1 = np.random.uniform(0, 1, (layer_in, layer_hid1))
weights_hid2 = np.random.uniform(0, 1,(layer_hid1, layer_hid2)) #сделал значение весов только положительные, из-за relu градиент был все время ноль и веса не обновлялись
weights_out  = np.random.uniform(0, 1,(layer_hid2, layer_out))

learning_rate = 0.0001 #при learning rate 0.001 модель перестает обучаться, как я понял из-за большого скачка на первой эпохе и потом отката в отрицательные значения для весов, а там уже relu
num_epochs = 350 #в данной модели после примерно 200 эпох значения ошибки начинают плавать около минимального значения примерно 42 

for epoch in range(1, num_epochs + 1):
    total_error = 0

    for i in range(len(inp)):
        layer_in = inp[i:i+1]
        true_out = true_prediction[i:i+1]

        hid1_raw = np.dot(layer_in, weights_hid1)
        hid1 = relu(hid1_raw)

        hid2_raw = np.dot(hid1, weights_hid2)
        hid2 = relu(hid2_raw)

        output = np.dot(hid2, weights_out)

        error = output - true_out
        total_error += np.sum(error ** 2)

        output_delta = error
        hid2_delta = output_delta.dot(weights_out.T) * relu_deriv(hid2_raw)
        hid1_delta = hid2_delta.dot(weights_hid2.T) * relu_deriv(hid1_raw)

        weights_out  -= learning_rate * hid2.T.dot(output_delta)
        weights_hid2 -= learning_rate * hid1.T.dot(hid2_delta)
        weights_hid1 -= learning_rate * layer_in.T.dot(hid1_delta)

    print(f"Epoch {epoch}, Error: {total_error:.4f}")

print("\nFinal predictions:")
for i in range(len(inp)):
    hid1 = relu(np.dot(inp[i:i+1], weights_hid1))
    hid2 = relu(np.dot(hid1, weights_hid2))
    pred = np.dot(hid2, weights_out)
    print(f"Input: {inp[i]}, Prediction: {pred[0][0]:.4f}, True prediction: {true_prediction[i][0]}")
