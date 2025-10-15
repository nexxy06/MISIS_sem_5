import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats


def read_data(filename):
    with open(filename, "r") as f:
        return [int(line.strip()) for line in f.readlines()]


def mean(data):
    return sum(data) / len(data)


def std(data, ddof=0):
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - ddof)
    return math.sqrt(variance)


def normal_cdf(x, mu=0, sigma=1):
    return (1 + math.erf((x - mu) / (sigma * math.sqrt(2)))) / 2


def main():
    gamma = 0.95
    delta = 3
    num_samples = 36

    ages = read_data("./Методы_статистического_анализа_данных/lab1/Москва_2021.txt")

    sigma_estimate = std(ages, ddof=0)

    # Определение объема выборки
    z = stats.norm.ppf((1 + gamma) / 2)  # z = 1.96 для γ = 0.95
    n_required = (z**2 * sigma_estimate**2) / delta**2
    n = int(math.ceil(n_required))

    print(f"Объем генеральной совокупности: {len(ages)}")
    print(f"Средний возраст: {mean(ages):.2f}")
    print(f"Стандартное отклонение: {sigma_estimate:.2f}")
    print(f"Объем выборки: {n}")
    print(f"z-значение (для объема выборки): {z:.3f}")

    # random.seed(42)
    sample_means = []

    for i in range(num_samples):
        sample = random.choices(ages, k=n)  
        sample_mean = mean(sample)
        sample_means.append(sample_mean)

    # Определение границ интервалов
    min_mean = math.floor(min(sample_means))
    max_mean = math.ceil(max(sample_means))
    interval_length = 1

    intervals = []
    current = min_mean
    while current < max_mean:
        intervals.append((current, current + interval_length))
        current += interval_length

    # Подсчет частот
    frequencies = []
    for interval in intervals:
        count = sum(
            1 for mean_val in sample_means if interval[0] <= mean_val < interval[1]
        )
        frequencies.append(count)

    relative_frequencies = [freq / len(sample_means) for freq in frequencies]

    print(f"\nИнтервальный ряд:")
    print("Интервал       Абс.частота Отн.частота")
    for i, interval in enumerate(intervals):
        print(
            f"{interval[0]:5.1f}-{interval[1]:5.1f}  {frequencies[i]:11d}  {relative_frequencies[i]:11.3f}"
        )


    x_positions = [interval[0] for interval in intervals]
    width = interval_length

    plt.figure(figsize=(12, 6))
    plt.bar(
        x_positions, relative_frequencies, width=width, alpha=0.7, edgecolor="black"
    )
    plt.xlabel("Выборочное среднее")
    plt.ylabel("Относительная частота")
    plt.title("Распределение выборочных средних")
    plt.grid(True, alpha=0.3)
    plt.xticks(x_positions, rotation=45)
    plt.tight_layout()
    plt.show()

    demo_sample = random.choices(ages, k=n)
    sample_mean = mean(demo_sample)
    sample_std = std(demo_sample, ddof=1)

    # Построение доверительного интервала для математического ожидания 
    # при неизвестной дисперсии генеральной совокупности
    t_student = stats.t.ppf((1 + gamma) / 2, df=n-1)  # df = n-1 степеней свободы

    margin = t_student * sample_std / math.sqrt(n)
    conf_interval = (sample_mean - margin, sample_mean + margin)

    print(f"\nДоверительный интервал (t-распределение Стьюдента):")
    print(f"Выборочное среднее: {sample_mean:.2f}")
    print(f"Исправленное СКО: {sample_std:.2f}")
    print(f"t-значение (Стьюдент, df={n-1}): {t_student:.3f}")
    print(f"Доверительный интервал: ({conf_interval[0]:.2f}, {conf_interval[1]:.2f})")
    print(f"Ширина интервала: {conf_interval[1] - conf_interval[0]:.2f}")
    print(f"Стандартное отклонение генеральной совокупности: {sigma_estimate:.2f}")

if __name__ == "__main__":
    main()
