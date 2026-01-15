import numpy as np


class Tensor(object):
    _next_id = 0

    def __init__(self, data, creators=None, operation=None, autograd=False, id=None):
        self.data = np.array(data)
        self.creators = creators
        self.operation_on_creation = operation
        self.autograd = autograd
        self.grad = None
        self.children = {}

        if id is None:
            self.id = Tensor._next_id
            Tensor._next_id += 1
        else:
            self.id = id

        if creators is not None:
            for parent in creators:
                parent.children[self.id] = parent.children.get(self.id, 0) + 1

    def __add__(self, other):
        if self.autograd and other.autograd:
            return Tensor(
                self.data + other.data,
                creators=[self, other],
                operation="+",
                autograd=True
            )
        return Tensor(self.data + other.data)

    def backward(self, grad=None, from_child=None):
        if not self.autograd:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data))

        if from_child is not None:
            self.children[from_child.id] -= 1

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad

        if self.creators is not None and (
            self._children_done() or from_child is None
        ):
            if self.operation_on_creation == "+":
                self.creators[0].backward(self.grad, self)
                self.creators[1].backward(self.grad, self)

    def _children_done(self):
        return all(v == 0 for v in self.children.values())

    def __str__(self):
        return str(self.data)


#Задание 1
print("Тест 1: проверка автоградиента")
print("=" * 50)

x1 = Tensor([1, 2, 3], autograd=True)
x2 = Tensor([1, 2, 3], autograd=True)
x3 = Tensor([1, 2, 3], autograd=True)

print("Входные данные:")
print("x1:", x1, "id:", x1.id)
print("x2:", x2, "id:", x2.id)
print("x3:", x3, "id:", x3.id)

s1 = x1 + x2
s2 = x2 + x3
out = s1 + s2

print("\nРезультаты:")
print("s1 =", s1)
print("s2 =", s2)
print("out =", out)

print("\nЗапуск backward с градиентом [4, 5, 3]")
out.backward(Tensor([4, 5, 3]))

print("\nГрадиенты:")
print("x1.grad =", x1.grad)
print("x2.grad =", x2.grad)
print("x3.grad =", x3.grad)

g1 = np.array([4, 5, 3])
g2 = np.array([8, 10, 6])
g3 = np.array([4, 5, 3])

if (x1.grad.data == g1).all() and (x2.grad.data == g2).all() and (x3.grad.data == g3).all():
    print("\nГрадиенты посчитаны верно")
else:
    print("\nОшибка в вычислении градиентов")


#Задание 2
print("\n" + "=" * 50)
print("Тест 2: проверка ID")

Tensor._next_id = 0

t1 = Tensor([1, 2, 3], autograd=True)
t2 = Tensor([4, 5, 6], autograd=True)
t3 = Tensor([7, 8, 9], autograd=True)

print("ID:", t1.id, t2.id, t3.id)

t4 = Tensor([10, 11, 12], autograd=True)
t5 = Tensor([13, 14, 15], autograd=True)

print("ID:", t4.id, t5.id)
print("Текущее значение _next_id:", Tensor._next_id)
