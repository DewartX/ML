import numpy as np

def relu(x):
    return (x > 0) * x

inp = np.array([
    [15, 10],
    [15, 15],
    [15, 20],
    [25, 10]
])

true_prediction = np.array([[10, 20, 15, 20]]).T

layer_in_size = inp.shape[1]
layer_hid1_size = 6
layer_hid2_size = 4
layer_out_size = 1

weights_hid1 = 2 * np.random.random((layer_in_size, layer_hid1_size)) - 1
weights_hid2 = 2 * np.random.random((layer_hid1_size, layer_hid2_size)) - 1
weights_out = 2 * np.random.random((layer_hid2_size, layer_out_size)) - 1

x = inp[0]

hid1 = relu(np.dot(x, weights_hid1))
hid2 = relu(np.dot(hid1, weights_hid2))
prediction = np.dot(hid2, weights_out)

print("Hidden layer 1:", hid1)
print("Hidden layer 2:", hid2)
print("Prediction:", prediction)
