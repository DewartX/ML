import numpy as np

arr = np.array([1,2,3,4,5])
print("Одномерный массив умноженный на 2")
print(arr*2)
print("=" * 50)

arr_21 = np.random.rand(3,3)
arr_22 = np.random.rand(3,3)

print("Перемноженные 2 случайных двумерных массива: \n", arr_21*arr_22)
print("=" * 50)

arr3 = np.random.randint(-10,11, size=10)
print("Одномерный массив из 10 случайных чисел \n", arr3)
print("=" * 50)
print("Четные числа из одномерного массива из 10 случайных чисел")
mask = arr3%2 == 0
print(arr3[mask])

print("Среднее значение массива:", arr3.mean())
print("Стандартное отклонение:", arr3.std())
print("Максимум:", arr3.max())
print("Минимум:", arr3.min())