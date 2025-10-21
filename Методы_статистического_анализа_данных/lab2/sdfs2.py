import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def read_data(filename):
    with open(filename, "r") as f:
        return [int(line.strip()) for line in f.readlines()]


def mean(data):
    return sum(data) / len(data)


def std(data, ddof=0):
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / (len(data) - ddof)
    return math.sqrt(variance)

def linspace(start, stop, num):
    step = (stop - start) / (num - 1)
    return [start + i * step for i in range(num)]

def norm_pdf(x, mu, sigma):
    """Функция плотности нормального распределения"""
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )

def main():
    gamma = 0.95
    delta = 3
    num_samples = 36

    ages = read_data("./Методы_статистического_анализа_данных/lab1/Москва_2021.txt")

    sigma_estimate = std(ages, ddof=0)

    # Определение объема выборки
    z = stats.norm.ppf((1 + gamma) / 2)
    n_required = (z**2 * sigma_estimate**2) / delta**2
    n = int(math.ceil(n_required))

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

    # Часть 1: Выравнивание статистического ряда и метод моментов
    print("\n" + "=" * 60)
    print("ЧАСТЬ 1: ВЫРАВНИВАНИЕ СТАТИСТИЧЕСКОГО РЯДА И АППРОКСИМАЦИЯ")
    print("=" * 60)

    # Вычисление среднего и стандартного отклонения для интервального ряда
    # Используем середины интервалов
    midpoints = [(interval[0] + interval[1]) / 2 for interval in intervals]

    # Точечные оценки методом моментов
    mu_estimate = sum(
        midpoints[i] * relative_frequencies[i] for i in range(len(intervals))
    )
    sigma_estimate_interval = math.sqrt(
        sum(
            relative_frequencies[i] * (midpoints[i] - mu_estimate) ** 2
            for i in range(len(intervals))
        )
    )

    print(f"Оценка математического ожидания (метод моментов): {mu_estimate:.3f}")
    print(
        f"Оценка стандартного отклонения (метод моментов): {sigma_estimate_interval:.3f}"
    )

    # Вычисление теоретических частот для нормального распределения
    theoretical_frequencies = []
    total_frequency = sum(frequencies)

    for interval in intervals:
        # Вероятность попадания в интервал для нормального распределения
        prob = stats.norm.cdf(
            interval[1], mu_estimate, sigma_estimate_interval
        ) - stats.norm.cdf(interval[0], mu_estimate, sigma_estimate_interval)
        theoretical_freq = prob * total_frequency
        theoretical_frequencies.append(theoretical_freq)

    # Относительные теоретические частоты
    theoretical_relative_freq = [
        freq / total_frequency for freq in theoretical_frequencies
    ]

    print(f"\nВыравнивание статистического ряда:")
    print("Интервал       Набл.част. Теор.част. Набл.отн.ч. Теор.отн.ч.")
    for i, interval in enumerate(intervals):
        print(
            f"{interval[0]:5.1f}-{interval[1]:5.1f}  {frequencies[i]:10d}  {theoretical_frequencies[i]:9.2f}  {relative_frequencies[i]:10.3f}  {theoretical_relative_freq[i]:9.3f}"
        )

    # Аппроксимация гистограммы кривой Гаусса
    # μ (математическое ожидание) = {mu_estimate:.4f}
    # σ (стандартное отклонение) = {sigma_estimate_interval:.4f}

    # Проверка качества аппроксимации (хи-квадрат)
    # chi_squared = 0
    # for i in range(len(frequencies)):
    #     if theoretical_frequencies[i] > 0:
    #         chi_squared += (
    #             frequencies[i] - theoretical_frequencies[i]
    #         ) ** 2 / theoretical_frequencies[i]

    # print(f"Статистика хи-квадрат для проверки аппроксимации: {chi_squared:.4f}")

    # Построение гистограммы с кривой Гаусса (аппроксимация)
    plt.figure(figsize=(14, 8))

    # Гистограмма наблюдаемых частот
    x_positions = [interval[0] for interval in intervals]
    width = interval_length

    plt.bar(
        x_positions,
        relative_frequencies,
        width=width,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
        label="Наблюдаемые относительные частоты",
    )

    # Аппроксимирующая кривая Гаусса
    x_min = min_mean
    x_max = max_mean
    x_curve = linspace(x_min, x_max, 1000)

    # Для правильной аппроксимации масштабируем плотность вероятности на ширину интервала
    y_curve = [norm_pdf(x, mu_estimate, sigma_estimate_interval) * width for x in x_curve]

    plt.plot(
        x_curve,
        y_curve,
        "r-",
        linewidth=3,
        label=f"Аппроксимирующая кривая Гаусса\nN({mu_estimate:.2f}, {sigma_estimate_interval:.2f}²)",
    )

    # Добавляем теоретические вероятности в виде точек
    plt.xlabel("Выборочное среднее", fontsize=12)
    plt.ylabel("Относительная частота / Плотность вероятности", fontsize=12)
    plt.title(
        "Аппроксимация распределения выборочных средних нормальным распределением",
        fontsize=14,
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(x_positions, rotation=45)
    plt.tight_layout()
    plt.show()

    # Часть 2: Доверительный интервал для одной выборки
    print("\n" + "=" * 60)
    print("ЧАСТЬ 2: ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ ВОЗРАСТА")
    print("=" * 60)

    # Берем одну случайную выборку из возрастов
    random_sample = random.choices(ages, k=n)
    sample_mean_age = mean(random_sample)
    sample_std_age = std(random_sample, ddof=1)  # Исправленное стандартное отклонение

    # Доверительный интервал с использованием t-распределения Стьюдента
    t_value = stats.t.ppf((1 + gamma) / 2, df=n - 1)
    margin_error = t_value * sample_std_age / math.sqrt(n)
    confidence_interval = (
        sample_mean_age - margin_error,
        sample_mean_age + margin_error,
    )

    print(f"Объем выборки: {n}")
    print(f"Точечная оценка математического ожидания: {sample_mean_age:.3f}")
    print(f"Исправленное стандартное отклонение: {sample_std_age:.3f}")
    print(f"Квантиль распределения Стьюдента (t-значение): {t_value:.3f}")
    print(f"Точность оценки (полуширина интервала): {margin_error:.3f}")
    print(
        f"Доверительный интервал ({gamma*100}%): ({confidence_interval[0]:.3f}, {confidence_interval[1]:.3f})"
    )
    print(
        f"Ширина доверительного интервала: {confidence_interval[1] - confidence_interval[0]:.3f}"
    )


if __name__ == "__main__":
    main()
