import numpy as np
from keras.src.datasets import mnist

def relu(x):
    return np.maximum(0, x)

def reluderiv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

train_images_count = 1000
test_images_count = 1000
pixels_per_image = 28 * 28
digits_num = 10
hidden_size = 100 #при уменьшении количества нейронов в скрытом слое в 2 раза, обучение идет чуть дольше, точность чуть уменьшается (по моим наблюдениям)
learning_rate = 0.0003
num_epoch = 200

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = x_train[:train_images_count].reshape(train_images_count, pixels_per_image) / 255.0
test_images = x_test[:test_images_count].reshape(test_images_count, pixels_per_image) / 255.0

def one_hot(labels, num_classes):
    out = np.zeros((len(labels), num_classes))
    for i, val in enumerate(labels):
        out[i, val] = 1
    return out

train_labels = one_hot(y_train[:train_images_count], digits_num)
test_labels = one_hot(y_test[:test_images_count], digits_num)

np.random.seed(2)
weight_hid = np.random.uniform(-0.1, 0.1, (pixels_per_image, hidden_size))  #диапазон весов
weight_out = np.random.uniform(-0.1, 0.1, (hidden_size, digits_num))

#Обучение
for epoch in range(num_epoch):
    correct_answers = 0
    for j in range(len(train_images)):
        layer_in = train_images[j:j+1]
        layer_hid = relu(np.dot(layer_in, weight_hid))
        layer_out = softmax(np.dot(layer_hid, weight_out))  # исправлено: softmax на выходе

        correct_answers += int(np.argmax(layer_out) == np.argmax(train_labels[j:j+1]))

        layer_out_delta = layer_out - train_labels[j:j+1]  # градиент кросс-энтропии
        layer_hid_delta = layer_out_delta.dot(weight_out.T) * reluderiv(layer_hid)

        weight_out -= learning_rate * layer_hid.T.dot(layer_out_delta)
        weight_hid -= learning_rate * layer_in.T.dot(layer_hid_delta)

    accuracy = correct_answers * 100 / len(train_images)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy:.2f}%")

print("-"*50)
print("Тестовые значения:")

#Тест
test_slice = 100
correct_test = 0

for j in range(test_slice):
    layer_in_test = test_images[j:j+1]
    
    layer_hid = relu(np.dot(layer_in_test, weight_hid))
    layer_out = np.dot(layer_hid, weight_out)

    # Softmax — превращаем логиты в вероятности
    exp_layer = np.exp(layer_out)
    layer_out_prob = exp_layer / np.sum(exp_layer, axis=1, keepdims=True)

    predicted_class = np.argmax(layer_out_prob)          # Выбираем индекс с максимальной вероятностью
    true_class = np.argmax(test_labels[j:j+1])

    correct_test += int(predicted_class == true_class)

    print(f"True: {true_class}, Predicted: {predicted_class}")

# Точность на выбранных тестовых изображениях
accuracy = correct_test / test_slice * 100
print(f"Test accuracy (first {test_slice} images): {accuracy:.2f}%")
