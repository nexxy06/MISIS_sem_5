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
z = 1.96  # квантиль нормального распределения
delta = 3
n = ceil((z * stddev / delta) ** 2)

samples = [random.choices(dataraw2, k=n) for _ in range(num_samples)]
sample_means = [sum(s) / n for s in samples]

a = floor(min(sample_means))
b = ceil(max(sample_means))
intervals = list(range(a, b))
freq = [0] * (b - a)
for mean in sample_means:
    idx = int(mean) - a
    if idx == len(freq):
        idx -= 1
    freq[idx] += 1
rel_freq = [f / len(sample_means) for f in freq]
intervals_full = [(intervals[i], intervals[i] + 1, rel_freq[i]) for i in range(len(intervals))]
midpoints = [(a + b) / 2 for a, b, f in intervals_full]

print("\nГистограмма частот по средним значениям выборок:")
for i, (left, right, rf) in enumerate(intervals_full):
    print(f"  Интервал [{left}, {right}): относительная частота = {rf:.3f}")

# Аппроксимация гистограммы кривой Гаусса
mean_sample_means = sum(sample_means) / len(sample_means)
std_sample_means = (sum((x - mean_sample_means) ** 2 for x in sample_means) / len(sample_means)) ** 0.5
bin_width = midpoints[1] - midpoints[0]
x_vals = [min(midpoints) - 1 + i for i in range(int((max(midpoints) - min(midpoints) + 2)))]
gauss_vals = [
    (1 / (std_sample_means * math.sqrt(2 * math.pi))) * math.exp(-((x - mean_sample_means) ** 2) / (2 * std_sample_means ** 2)) * bin_width
    for x in x_vals
]

print("\nПараметры аппроксимирующей кривой Гаусса:")
print(f"  Оценка мат. ожидания: {mean_sample_means:.3f}")
print(f"  Оценка стандартного отклонения: {std_sample_means:.3f}")

plt.bar(midpoints, rel_freq, label="Гистограмма")
plt.plot(x_vals, gauss_vals, 'r', label="Кривая Гаусса")
plt.legend()
plt.title("Гистограмма и аппроксимирующая кривая Гаусса")
plt.xlabel("Средний возраст")
plt.ylabel("Относительная частота")

# Доверительный интервал для одной выборки
toanalyse = samples[0]
mean = sum(toanalyse) / len(toanalyse)
alpha = 1 - gamma
dfree = len(toanalyse) - 1
sample_std = (sum((x - mean) ** 2 for x in toanalyse) / dfree) ** 0.5
t_crit = t.ppf(1 - alpha / 2, dfree)
margin = t_crit * sample_std / (len(toanalyse) ** 0.5)
trust_interval_sample = (float(mean - margin), float(mean + margin))
accuracy = z * stddev / (n ** 0.5)

print("\n Доверительный интервал для математического ожидания (одна выборка):")
print(f"  Точечная оценка: {mean:.3f}")
print(f"  Интервал: [{trust_interval_sample[0]:.3f}, {trust_interval_sample[1]:.3f}]")
print(f"  Точность: {accuracy:.3f}")
print(f"  Квантиль Стьюдента: {t_crit:.3f}")
dfree = len(toanalyse) - 1
sample_std = (sum((x - mean) ** 2 for x in toanalyse) / dfree) ** 0.5
t_crit = t.ppf(1 - alpha / 2, dfree)
print('t:', t_crit)
margin = t_crit * sample_std / (len(toanalyse) ** 0.5)
trust_interval_sample = (float(mean - margin), float(mean + margin))
print('Доверительный интервал:', trust_interval_sample)
accuracy = z * stddev / (n ** 0.5)
print('Точность выборочного среднего', accuracy)

plt.plot(x_vals, [y for y in gauss_vals], 'r')
plt.bar(midpoints, rel_freq)
plt.show()
