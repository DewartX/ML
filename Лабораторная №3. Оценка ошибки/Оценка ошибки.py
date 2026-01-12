import numpy as np

def neural_networks(inp, weights):
	return inp.dot(weights)

def get_error(true_prediction, prediction):
	return (true_prediction - prediction) ** 2

weights = np.array([0.2, 0.3])
input = np.array([150,40])
step = 0.001

true_pred = 50

while True:
	pred = neural_networks(input, weights)
	loss = get_error(true_pred,pred)
	if loss > 0.001:
		if pred > true_pred:
			weights -= step
		elif pred < true_pred:
			weights += step
		else:
			break
	elif loss <= 0.001:
		break

print("Получившиеся веса:", weights)
print("Loss:", loss)