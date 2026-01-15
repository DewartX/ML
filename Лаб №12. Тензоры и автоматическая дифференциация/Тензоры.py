import numpy as np


class Tensor(object):
    def __init__(self, data, creators=None, operation_on_creation=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation_on_creation
        self.grad = None

    def __add__(self, other):
        return Tensor(
            self.data + other.data,
            creators=[self, other],
            operation_on_creation="+"
        )

    def backward(self, grad):
        self.grad = grad

        if self.operation_on_creation == "+":
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)

    def __str__(self):
        return str(self.data)


t_1 = Tensor([3, 15, 10])
t_2 = Tensor([5, 6, 7])
print(t_1, t_2)

t_3 = t_1 + t_2
print(t_3)

t_3.backward(Tensor([1, 2, 3]))
print(t_1.grad, t_2.grad)
print(t_3.operation_on_creation)

a_1 = Tensor([1, 2, 3])
a_2 = Tensor([1, 2, 3])
a_3 = Tensor([1, 2, 3])
a_4 = Tensor([1, 2, 3])

a_add_1 = a_1 + a_2
a_add_2 = a_3 + a_4
a_add_3 = a_add_1 + a_add_2

a_add_3.backward(Tensor([4, 5, 3]))
print(a_1.grad)
