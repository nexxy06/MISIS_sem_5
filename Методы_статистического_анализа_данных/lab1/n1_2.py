from matplotlib import pyplot as plt

with open("Москва_2021.txt", "r") as infile:
    dataraw = infile.read()
data = [(int(item), dataraw.count(item)) for item in sorted(set(dataraw.splitlines()))]
# print(data)

# Расчет основных статистических показателей
average = sum(k[0] * k[1] for k in data) / sum(k[1] for k in data)
variance = sum((item[0] - average) ** 2 * item[1] for item in data) / sum(
    item[1] for item in data
)
stddev = variance**0.5

mode = max(data, key=lambda x: x[1])

median = 0
total = 0
for value, freq in sorted(data):
    total += freq
    if total >= sum(k[1] for k in data) / 2:
        median = value
        break

range_val = max(k[0] for k in data) - min(k[0] for k in data)
variation_coefficient = stddev / average

# Расчет асимметрии и эксцесса
assymetry = (
    sum(((k[0] - average) ** 3) * k[1] for k in data)
    / sum(k[1] for k in data)
    / (stddev**3)
)
excess = (
    sum(((k[0] - average) ** 4) * k[1] for k in data)
    / sum(k[1] for k in data)
    / (stddev**4)
    - 3
)

print(f"Ассиметрия {assymetry:.3f}")
print(f"Эксцесс {excess:.3f}")

threshold = 0.5
three_sigma_rule = {1: 68.3, 2: 95.4, 3: 99.7}
rule_applies = True
total = sum(item[1] for item in data)

for i in range(1, 4):
    low = average - i * stddev
    high = average + i * stddev

    in_interval = [item for item in data if low <= item[0] <= high]
    count_in_interval = sum(item[1] for item in in_interval)
    percentage_in_interval = count_in_interval / total * 100

    print(f"Интервал [{low:.3f}, {high:.3f}]")
    print(f"Процент реализаций в интервале {i} сигм: {percentage_in_interval:.3f}%")

    if abs(three_sigma_rule[i] - percentage_in_interval) > threshold:
        print(
            f"В этом интервале правило 3 сигм НЕ соблюдается, ошибка на {abs(three_sigma_rule[i]-percentage_in_interval):.3f}%"
        )
        rule_applies = False
    else:
        print("В этом интервале правило 3 сигм соблюдается")


if rule_applies:
    print("Правило 3 сигм соблюдается для всех интервалов")
else:
    print("Правило 3 сигм НЕ соблюдается => распределение не нормальное")

# Построение кумулятивной кривой
data_cum = []
cum = 0
for value, freq in sorted(data):
    cum += freq
    data_cum.append((value, cum / total))

# print(data_cum)

values, freqs = zip(*data_cum)
plt.plot(values, freqs)
plt.title("Кумулятивная кривая распределения")
plt.xlabel("Значения")
plt.ylabel("Накопленная частота")
plt.grid(True)
plt.show()
