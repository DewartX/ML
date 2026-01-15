import numpy as np


class Tensor(object):
    
    id_count = 0

    def __init__(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.init(data, creators, operation_on_creation, autograd, id)

    def init(self, data, creators=None, operation_on_creation=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None
        self.autograd = autograd
        self.children = {}
        
        if id is None:
            self.__class__.id_count += 1
            self.id = self.__class__.id_count
            
        if self.creators is not None:
            for creator in creators:
                if self.id not in creator.children:
                    creator.children[self.id] = 1
                else:
                    creator.children[self.id] += 1

    def backward(self, grad=None, grad_origin=None):
        if self.autograd:
            if grad is None:    
                grad = Tensor(np.ones_like(self.data))
            if grad_origin is not None:
                if self.children[grad_origin.id] > 0:
                    self.children[grad_origin.id] -= 1
                            
        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        if self.creators is not None and (self.check_grads_from_children() or grad_origin is None):
                if self.operation_on_creation == "+":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad, self)
                    
                elif self.operation_on_creation == "-1":
                    self.creators[0].backward(self.grad.__neg__(), self)
                    
                elif self.operation_on_creation == "-":
                    self.creators[0].backward(self.grad, self)
                    self.creators[1].backward(self.grad.__neg__(), self)
                    
                elif self.operation_on_creation == "*":
                    new_grad = self.grad * self.creators[1]
                    self.creators[0].backward(new_grad, self)
                    new_grad = self.grad * self.creators[0]
                    self.creators[1].backward(new_grad, self)
                    
                elif "sum" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.expand(axis, self.creators[0].data.shape[axis]), self)
                    
                elif "expand" in self.operation_on_creation:
                    axis = int(self.operation_on_creation.split("_")[1])
                    self.creators[0].backward(self.grad.sum(axis), self)
                    
                elif self.operation_on_creation == "transpose":
                    self.creators[0].backward(self.grad.transpose(), self)
                    
                elif self.operation_on_creation == "dot":
                    temp = self.grad.dot(self.creators[1].transpose())
                    self.creators[0].backward(temp, self)
                    temp = self.creators[0].transpose().dot(self.grad)
                    self.creators[1].backward(temp, self)

                elif self.operation_on_creation == "sigmoid":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (self*(temp - self)), self)

                elif self.operation_on_creation == "tanh":
                    temp = Tensor(np.ones_like(self.grad.data))
                    self.creators[0].backward(self.grad * (temp - self * self), self)

                elif self.operation_on_creation == "relu":
                    temp = self.grad * Tensor((self.creators[0].data > 0) * 1.0)
                    self.creators[0].backward(temp, self)
                
                elif self.operation_on_creation == "softmax":
                    self.creators[0].backward(Tensor(self.grad.data), self)
                
                elif self.operation_on_creation == "pow":
                    num = self.creators[0]
                    power = self.creators[1].data
                    new_grad = self.grad * Tensor(power * (num.data ** (power - 1)))
                    num.backward(new_grad, self)
                                    
    def check_grads_from_children(self):
        for id in self.children:
            if self.children[id] != 0:
                return False
        return True
            
    def add(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data + other.data, [self, other], "+", True)
        return Tensor(self.data + other.data)

    def __add__(self, other):
        return self.add(other)
    
    def __str__(self):
        return str(self.data)

    # substract
    def __sub__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data - other.data, [self,other], "-", True)
        return Tensor(self.data - other.data)

    # negative
    def __neg__(self):
        if self.autograd:
            return Tensor(self.data * (-1), [self], "-1", True)
        return Tensor(self.data * (-1))

    # multiple
    def __mul__(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data * other.data, [self, other], "*", True)
        return Tensor(self.data * other.data)

    # sum
    def sum(self, axis):
        if self.autograd:
            return Tensor(self.data.sum(axis), [self], "sum_" + str(axis), True)
        return Tensor(self.data.sum(axis))
    
    # expand
    def expand(self, axis, count_copies):
        transpose = list(range(0, len(self.data.shape)))
        transpose.insert(axis, len(self.data.shape))
        expand_shape = list(self.data.shape) + [count_copies]
        expand_data = (self.data.repeat(count_copies).reshape(expand_shape))
        expand_data = expand_data.transpose(transpose)
        if self.autograd:
            return Tensor(expand_data, [self], "expand_" + str(axis), True)
        return Tensor(expand_data)

    # dot
    def dot(self, other):
        if self.autograd and other.autograd:
            return Tensor(self.data.dot(other.data), [self, other], "dot", True)
        return Tensor(self.data.dot(other.data))

    # transpose
    def transpose(self):
        if self.autograd:
            return Tensor(self.data.transpose(), [self], "transpose", True)
        return Tensor(self.data.transpose())

    # sigmoind
    def sigmoid(self):
        if self.autograd:
            return Tensor(1/(1+np.exp(-self.data)), [self], "sigmoid",True)
        return Tensor(1/(1+np.exp(-self.data)))
    
    # tanh
    def tanh(self):
        if self.autograd:
            return Tensor(np.tanh(self.data), [self], "tanh",True)
        return Tensor(np.tanh(self.data))

    # relu
    def relu(self):  
        if self.autograd:
            return Tensor((self.data > 0) * self.data, [self], "relu",True)
        return (self.data > 0) * self.data
    
    # repr
    def __repr__(self):
        return str(self.data.__repr__())
    
    # softmax
    def softmax(self):
        exp = np.exp(self.data)
        exp = exp / np.sum(exp, axis=1, keepdims=True)
        if self.autograd:
            return Tensor(exp, [self], "softmax", True)
        return Tensor(exp)
    
    def __pow__(self, power):
        if self.autograd:
            power_tensor = Tensor(power, autograd=False)
            return Tensor(self.data ** power, [self, power_tensor], "pow", True)
        return Tensor(self.data ** power)



class SGD(object):
    def __init__(self, weights, learning_rate = 0.01):
        self.weights = weights
        self.learning_rate = learning_rate

    def step(self):
        for weight in self.weights:
            weight.data -= self.learning_rate * weight.grad.data
            weight.grad.data *=0


class Layer(object):
    def __init__(self):
        self.parameters = []

    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, input_count, output_count):
        super().__init__()

        weight = np.random.randn(input_count, output_count) * np.sqrt(2.0/input_count)

        self.weight = Tensor(weight, autograd=True)
        self.bias = Tensor(np.zeros(output_count), autograd=True)
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, inp):
        return inp.dot(self.weight) + self.bias.expand(0, len(inp.data))


class Sequential(Layer):
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp
    
    def get_parameters(self):
        params = []
        for layer in self.layers:
            params += layer.get_parameters()
        return params

class Sigmoid(Layer):
    def forward(self, inp):
        return inp.sigmoid()

class Tanh(Layer):
    def forward(self, inp):
        return inp.tanh()

class MSELoss(Layer):
    def forward(self, prediction, true_prediction):
        diff = prediction - true_prediction
        return (diff * diff).sum(0) * Tensor(1.0 / prediction.data.shape[0], autograd=True)

class Softmax(Layer):
    def forward(self, inp):
        return inp.softmax()

class RMSELoss(Layer):
    def forward(self, prediction, true_prediction):
        return MSELoss().forward(prediction, true_prediction) ** 0.5


'''задания 1 - 3

np.random.seed(0)

x = Tensor([
    [1, 2, 3],
    [2, 3, 4],
    [3, 4, 5],
    [1, 3, 5],
], autograd=True)

y = Tensor([
    [6],
    [24],
    [60],
    [15],
], autograd=True)

model = Sequential([
    Linear(3, 10),
    Tanh(),
    Linear(10, 1)
])'''


'''# 40.32678815 - предсказание с MSELoss
loss = MSELoss()
sgd = SGD(model.get_parameters(), learning_rate=0.001)

for epoch in range(1500):
    preds = model.forward(x)
    error = loss.forward(preds, y)

    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {error}")

test = Tensor([[2, 5, 4]])
print("Prediction:", model.forward(test))
print("True:", 2 * 5 * 4)
'''

'''# 40.0283695 - предсказание с RMSELoss но RMSE даёт большой градиент из-за корня поэтому нужно больше эпох и больший learning_rate
loss = RMSELoss()
sgd = SGD(model.get_parameters(), learning_rate=0.01)

for epoch in range(2000):
    preds = model.forward(x)
    error = loss.forward(preds, y)

    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {error}")

test = Tensor([[2, 5, 4]])
print("Prediction:", model.forward(test))
print("True:", 2 * 5 * 4)'''


# задания 4, 5

x = Tensor([
    [80, 25],
    [90, 30],
    [85, 35],
    [45, 22],
    [55, 28],
    [50, 35],
], autograd=True)

y = Tensor([
    [1, 0],  # муж
    [1, 0],  # муж
    [1, 0],  # муж
    [0, 1],  # жен
    [0, 1],  # жен
    [0, 1],  # жен
], autograd=True)


model = Sequential([
    Linear(2, 8),
    Tanh(),
    Linear(8, 2),
    Sigmoid()
])

loss = MSELoss()
sgd = SGD(model.get_parameters(), learning_rate=0.01)


for epoch in range(10000):
    preds = model.forward(x)
    error = loss.forward(preds, y)

    error.backward(Tensor(np.ones_like(error.data)))
    sgd.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss {error}")


def predict(weight, age):
    out = model.forward(Tensor([[weight, age]]))
    return out.data

print(predict(80, 25))
print(predict(55, 30))
