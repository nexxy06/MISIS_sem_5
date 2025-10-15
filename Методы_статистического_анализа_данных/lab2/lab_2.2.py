import math
import random
import matplotlib.pyplot as plt 
from scipy.stats import t
listik = [int(line.strip()) for line in open('./Методы_статистического_анализа_данных/lab1/Москва_2021.txt', encoding='utf-8').readlines()]

y = 0.95  # надёжность
table_error_koef = 1.96  # кэф ошибки по таблице
precision = 3  # δ = 3 года 


def dispersion(data):
    freq = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1

    mean = sum(x * f for x, f in zip(freq.keys(), freq.values())) / sum(freq.values())
    return sum(f * (x - mean) ** 2 for x, f in zip(freq.keys(), freq.values())) / sum(freq.values())


def SKO(dispersion_value):
    return dispersion_value ** 0.5


def selection_size_repeatable(table_error_koef, precision, sigma):
    return math.ceil((table_error_koef * sigma / precision) ** 2)


sigma = SKO(dispersion(listik))
n = selection_size_repeatable(table_error_koef, precision, sigma)

def dispersion_fixed(data):
    freq = {}
    for x in data:
        freq[x] = freq.get(x, 0) + 1
    
    mean = sum(x * f for x, f in zip(freq.keys(), freq.values())) / sum(freq.values())
    return sum(f * (x - mean) ** 2 for x, f in zip(freq.keys(), freq.values())) / (sum(freq.values()) - 1)


sigma_fixed = SKO(dispersion_fixed(listik))



selections_count = 36


def generate_selections(selection_size, selections_count, listik):
    selections = []
    for _ in range(selections_count):
        selection = [random.choice(listik) for _ in range(selection_size)]
        selections.append(selection)
    return selections

selections = generate_selections(n, selections_count, listik)

means = [(sum(sel) / len(sel)) for sel in selections]


def interval(data, interval_length=1):
    min_val, max_val = math.floor(min(data)), math.ceil(max(data))
    h = interval_length
    intervals = []
    lower = min_val
    while lower <= max_val:
        upper = lower + h
        count = sum(1 for x in data if lower <= x < upper)
        intervals.append([lower, upper, count])
        lower = upper

    print("\n -- Интервальный ряд --")
    total = len(data)
    for a, b, f in intervals:
        rel_freq = f / total
        print(f"[{a}; {b}) : {f} (отн. частота {rel_freq:.3f})")
    return intervals


intervals = interval(means, interval_length=1)


midpoints = [(a + b) / 2 for a, b, f in intervals]
rel_freqs = [f / len(means) for a, b, f in intervals]
width = 1  


sample_mean = sum(means) / len(means)
sample_var = sum((x - sample_mean) ** 2 for x in means) / len(means)
sample_sigma = math.sqrt(sample_var)

print(f"\nТочечные оценки:")
print(f"Мат. ожидание (μ) = {sample_mean:.3f}")
print(f"Дисперсия (σ²) = {sample_var:.3f}")
print(f"СКО (σ) = {sample_sigma:.3f}")



plt.bar(midpoints, rel_freqs, width=width, edgecolor="black", alpha=0.6, label="Гистограмма")


x_vals = [min(midpoints) - 1 + i * 0.1 for i in range(int((max(midpoints) - min(midpoints) + 2) * 10))]
gauss_vals = [
    (1 / (sample_sigma * math.sqrt(2 * math.pi))) * math.exp(-((x - sample_mean) ** 2) / (2 * sample_sigma ** 2))
    for x in x_vals
]

gauss_scaled = [y * width for y in gauss_vals]


mean_of_means = sum(means) / len(means)


delta = table_error_koef * sigma / math.sqrt(n)

t_val = table_error_koef  
lower_bound = mean_of_means - t_val * sigma_fixed / math.sqrt(n)
upper_bound = mean_of_means + t_val * sigma_fixed / math.sqrt(n)

alpha = 1 - y
t_value = t.ppf(1 - alpha/2, df=n-1)


print(f"\n t-критерий Стьюдента: {t_value}")

print(f"\n Погрешность выборочного среднего: {delta:.6f}")
print(f"Итого доверительный интервал: [{mean_of_means - delta:.3f} ; {mean_of_means + delta:.3f}]")


plt.plot(x_vals, gauss_scaled, "r-", linewidth=2, label="Кривая Гаусса")

plt.xlabel("Значения выборочных средних")
plt.ylabel("Относительная частота")
plt.title("Гистограмма и аппроксимация нормальным распределением")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()