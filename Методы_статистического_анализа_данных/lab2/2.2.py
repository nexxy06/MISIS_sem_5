import random
import math
from math import ceil, floor
from matplotlib import pyplot as plt
from scipy.stats import t


with open('./Методы_статистического_анализа_данных/lab1/Москва_2021.txt', 'r') as infile:
    dataraw = infile.read()
data = [(int(item), dataraw.count(item)) for item in sorted(set(dataraw.splitlines()))]
dataraw2 = [int(item) for item in dataraw.splitlines()]

# Точечные оценки параметров методом моментов
average = sum(k[0] * k[1] for k in data) / sum(k[1] for k in data)
variance = sum((item[0] - average) ** 2 * item[1] for item in data) / sum(item[1] for item in data)
stddev = variance ** 0.5
mode = max(data, key=lambda x: x[1])[0]

median = 0
total = 0
for value, freq in sorted(data):
    total += freq
    if total >= sum(k[1] for k in data) / 2:
        median = value
        break

# Размах и коэф вариации 
range_val = max(k[0] for k in data) - min(k[0] for k in data)
variation_coefficient = stddev / average

print("Точечные оценки параметров распределения:")
print(f"  Математическое ожидание (среднее): {average:.3f}")
print(f"  Дисперсия: {variance:.3f}")
print(f"  Стандартное отклонение: {stddev:.3f}")
print(f"  Мода: {mode}")
print(f"  Медиана: {median}")
print(f"  Диапазон: {range_val}")
print(f"  Коэффициент вариации: {variation_coefficient:.3f}")

# Генерация случайных выборок и построение гистограммы
num_samples = 36
gamma = 0.95
z = 1.96
delta = 3
# Объём выборки
n = ceil((z * stddev / delta) ** 2)


# Генерация 36 случайных выборок объема n
samples = [random.choices(dataraw2, k=n) for _ in range(num_samples)]

# Вычисление выборочных средних для каждой выборки
sample_means = [sum(s) / n for s in samples]

# Построение интервального ряда для выборочных средних
a = floor(min(sample_means))
b = ceil(max(sample_means))
intervals = list(range(a, b))
freq = [0] * (b - a)
for mean in sample_means:
    idx = int(mean) - a
    if idx == len(freq):
        idx -= 1
    freq[idx] += 1

# Вычисление относительных частот
rel_freq = [f / len(sample_means) for f in freq]
intervals_full = [(intervals[i], intervals[i] + 1, rel_freq[i]) for i in range(len(intervals))]
midpoints = [(a + b) / 2 for a, b, f in intervals_full]

print("\nГистограмма частот по средним значениям выборок:")
for i, (left, right, rf) in enumerate(intervals_full):
    print(f"  Интервал [{left}, {right}): относительная частота = {rf:.3f}")

# Аппроксимация гистограммы кривой Гаусса
mean_sample_means = sum(sample_means) / len(sample_means)
std_sample_means = (sum((x - mean_sample_means) ** 2 for x in sample_means) / len(sample_means)) ** 0.5    # Выборочное СКО
bin_width = midpoints[1] - midpoints[0]

# увеличить количество точек для построения кривой (только визуально, не влияет на вычисления)
smooth_points = 400  # можно менять для более/менее плавной линии
x_min = min(midpoints) - 1
x_max = max(midpoints) + 1
x_vals = [x_min + i * (x_max - x_min) / (smooth_points - 1) for i in range(smooth_points)]

# Вычисление значений плотности нормального распределения
gauss_vals = [
    (1 / (std_sample_means * math.sqrt(2 * math.pi))) * math.exp(-((x - mean_sample_means) ** 2) / (2 * std_sample_means ** 2)) * bin_width
    for x in x_vals
]

print("\nПараметры аппроксимирующей кривой Гаусса:")
print(f"  Оценка мат. ожидания: {mean_sample_means:.3f}")
print(f"  Оценка стандартного отклонения: {std_sample_means:.3f}")

plt.bar(midpoints, rel_freq, label="Гистограмма", alpha=0.7, width=1.0)
plt.plot(x_vals, gauss_vals, 'r', label="Кривая Гаусса")
plt.legend()
plt.title("Гистограмма и аппроксимирующая кривая Гаусса")
plt.xlabel("Средний возраст")
plt.ylabel("Относительная частота")
plt.show()

# Доверительный интервал для одной выборки
first_vib = samples[0]
mean = sum(first_vib) / len(first_vib)
# Параметры для t-распределения Стьюдента
alpha = 1 - gamma
dfree = len(first_vib) - 1

sample_std = (sum((x - mean) ** 2 for x in first_vib) / dfree) ** 0.5
t_crit = t.ppf(1 - alpha / 2, dfree)
margin = t_crit * sample_std / (len(first_vib) ** 0.5)
trust_interval_sample = (float(mean - margin), float(mean + margin))
accuracy = z * stddev / (n ** 0.5)

print("\n Доверительный интервал для математического ожидания (одна выборка):")
print(f"  Точечная оценка: {mean:.3f}")
print(f"  Интервал: [{trust_interval_sample[0]:.3f}, {trust_interval_sample[1]:.3f}]")
print(f"  Точность: {accuracy:.3f}")
print(f"  Квантиль Стьюдента: {t_crit:.3f}")

