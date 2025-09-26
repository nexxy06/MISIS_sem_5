import sys
import os
import math
import random
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '../lab1'))
# Импорт функций из ваших файлов
from n1_1 import mean, variance, std_dev, load_file


def calculate_assymetry(data, average, stddev):
    """Расчет асимметрии (из 1.2.py)"""
    total = sum(item[1] for item in data)
    return sum(((k[0] - average) ** 3) * k[1] for k in data) / total / (stddev**3)


def calculate_excess(data, average, stddev):
    """Расчет эксцесса (из 1.2.py)"""
    total = sum(item[1] for item in data)
    return sum(((k[0] - average) ** 4) * k[1] for k in data) / total / (stddev**4) - 3


def calculate_sample_size(sigma, delta, gamma):
    """
    n = t²σ²/δ²
    """
    # Определяем t по таблице Лапласа для заданной надежности
    if gamma == 0.95:
        t = 1.96
    elif gamma == 0.99:
        t = 2.58
    elif gamma == 0.999:
        t = 3.29
    else:
        t = 2.0

    n = (t**2 * sigma**2) / delta**2
    return math.ceil(n)


def generate_samples(data, sample_size, num_samples):
    """Генерация выборок и вычисление выборочных средних"""
    sample_means = []

    for _ in range(num_samples):
        # Повторная выборка (с возвращением)
        sample = random.choices(data, k=sample_size)
        sample_mean = mean(sample)
        sample_means.append(sample_mean)

    return sample_means


def create_interval_series(sample_means, interval_length=1):
    """Создание интервального ряда распределения"""
    min_mean = math.floor(min(sample_means))
    max_mean = math.ceil(max(sample_means))

    # Создание интервалов
    intervals = []
    current = min_mean
    while current <= max_mean:
        intervals.append((current, current + interval_length))
        current += interval_length

    # Подсчет частот
    frequencies = []
    for interval in intervals:
        count = 0
        for mean_val in sample_means:
            if interval[0] <= mean_val < interval[1]:
                count += 1
        frequencies.append(count)

    # Относительные частоты
    total_samples = len(sample_means)
    relative_frequencies = [freq / total_samples for freq in frequencies]

    return intervals, frequencies, relative_frequencies


def print_interval_series(intervals, frequencies, relative_frequencies):
    """Вывод интервального ряда в табличном виде"""
    print("\n" + "=" * 60)
    print("ИНТЕРВАЛЬНЫЙ РЯД РАСПРЕДЕЛЕНИЯ ВЫБОРОЧНЫХ СРЕДНИХ")
    print("=" * 60)
    print(f"{'Интервал':<15} {'Абс. частота':<12} {'Отн. частота':<12}")
    print("-" * 40)

    for i, interval in enumerate(intervals):
        abs_freq = frequencies[i]
        rel_freq = relative_frequencies[i]
        print(f"{interval[0]:.1f}-{interval[1]:.1f}{abs_freq:>12}{rel_freq:>12.3f}")


def plot_histogram(intervals, relative_frequencies):
    """Построение гистограммы относительных частот"""
    # Центры интервалов для позиционирования столбцов
    interval_centers = [(interval[0] + interval[1]) / 2 for interval in intervals]
    interval_widths = [interval[1] - interval[0] for interval in intervals]

    plt.figure(figsize=(12, 6))
    plt.bar(
        interval_centers,
        relative_frequencies,
        width=0.8,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    plt.xlabel("Выборочная средняя (возраст, лет)")
    plt.ylabel("Относительная частота")
    plt.title("Распределение выборочных средних возраста преступников")
    plt.grid(axis="y", alpha=0.3)

    # Добавление значений на столбцы
    for i, (center, freq) in enumerate(zip(interval_centers, relative_frequencies)):
        if freq > 0:
            plt.text(
                center,
                freq + 0.005,
                f"{freq:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.xticks(interval_centers, [f"{center:.1f}" for center in interval_centers])
    plt.tight_layout()
    plt.show()


def main():
    # 1. Загрузка и анализ генеральной совокупности
    print("АНАЛИЗ ГЕНЕРАЛЬНОЙ СОВОКУПНОСТИ")
    data = load_file("Москва_2021.txt")

    pop_mean = mean(data)
    pop_var = variance(data, pop_mean)
    pop_std = std_dev(pop_var)

    print(f"Объем генеральной совокупности: {len(data)}")
    print(f"Средний возраст: {pop_mean:.2f} лет")
    print(f"Дисперсия: {pop_var:.2f}")
    print(f"Стандартное отклонение: {pop_std:.2f} лет")

    # 2. Расчет объема выборки
    print("\n=== РАСЧЕТ ОБЪЕМА ВЫБОРКИ ===")
    gamma = 0.95  # надежность
    delta = 3  # точность (годы)

    sample_size = calculate_sample_size(pop_std, delta, gamma)
    print(f"Надежность (γ): {gamma}")
    print(f"Точность (δ): {delta} лет")
    print(f"Стандартное отклонение генеральной совокупности (σ): {pop_std:.2f} лет")
    print(f"Необходимый объем выборки: {sample_size}")

    # 3. Генерация 36 выборок
    print("\n=== ГЕНЕРАЦИЯ ВЫБОРОК ===")
    num_samples = 36
    random.seed(42)  # для воспроизводимости

    sample_means = generate_samples(data, sample_size, num_samples)

    print(f"Количество выборок: {num_samples}")
    print(f"Объем каждой выборки: {sample_size}")
    print(f"Минимальная выборочная средняя: {min(sample_means):.2f}")
    print(f"Максимальная выборочная средняя: {max(sample_means):.2f}")
    print(f"Среднее выборочных средних: {mean(sample_means):.2f}")

    # 4. Построение интервального ряда
    intervals, frequencies, relative_frequencies = create_interval_series(sample_means)
    print_interval_series(intervals, frequencies, relative_frequencies)

    # 5. Построение гистограммы
    print("\n=== ПОСТРОЕНИЕ ГИСТОГРАММЫ ===")
    plot_histogram(intervals, relative_frequencies)

    # 6. Проверка точности оценки
    print("\n=== ПРОВЕРКА ТОЧНОСТИ ОЦЕНКИ ===")
    sample_means_mean = mean(sample_means)
    error = abs(sample_means_mean - pop_mean)

    print(f"Истинное среднее генеральной совокупности: {pop_mean:.2f}")
    print(f"Среднее выборочных средних: {sample_means_mean:.2f}")
    print(f"Ошибка оценки: {error:.2f} лет")
    print(f"Требуемая точность: {delta} лет")
    print(f"Условие выполняется: {error <= delta}")


if __name__ == "__main__":
    main()
