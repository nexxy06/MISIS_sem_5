import math
import matplotlib.pyplot as plt

k = 7


def load_file(fname):
    d = []
    with open("Москва_2021.txt", "r") as f:
        for line in f:
            if line.strip():
                d.append(int(line.strip()))
    return d


def make_discrete_series(d):
    freq = {}
    for x in d:
        freq[x] = freq.get(x, 0) + 1
    series = []
    for x in sorted(freq.keys()):
        series.append({"x": x, "n": freq[x]})
    return series


def make_interval_series(d):
    global k
    x_min = min(d)
    x_max = max(d)

    h = math.ceil((x_max - x_min) / k)
    x_max = x_min + k * h
    print(x_min, x_max)
    intervals = []
    for i in range(k):
        a = x_min + i * h
        b = x_min + (i + 1) * h
        intervals.append((math.ceil(a), math.ceil(b)))
    print(intervals)

    n_i = []
    mid = []
    s_i = 0
    cumul = []

    for a, b in intervals:
        cnt = 0
        for x in d:
            if a <= x < b:
                cnt += 1
        n_i.append(cnt)
        mid.append((a + b) / 2)
        s_i += cnt
        cumul.append(s_i)

    res = []
    for i, (a, b) in enumerate(intervals):
        res.append(
            {"intv": f"{a:.1f}-{b:.1f}", "mid": mid[i], "n": n_i[i], "s": cumul[i]}
        )
    return res, h, x_max


def mean(d):
    return sum(d) / len(d)


def variance(d, m):
    return sum((x - m) ** 2 for x in d) / len(d)


def std_dev(var):
    return math.sqrt(var)


def mode_and_freq(d):
    freq = {}
    for x in d:
        freq[x] = freq.get(x, 0) + 1
    max_f = max(freq.values())
    modes = [x for x, f in freq.items() if f == max_f]
    return modes[0], max_f


def median(d):
    s = sorted(d)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    else:
        return (s[n // 2 - 1] + s[n // 2]) / 2


def range_val(d):
    return max(d) - min(d)


def cv(s, m):
    return (s / m) * 100 if m != 0 else 0


def calc_interval_mean(series):
    total_n = sum(item["n"] for item in series)
    weighted_sum = sum(item["mid"] * item["n"] for item in series)
    return weighted_sum / total_n


def calc_interval_variance(series, mean_val):
    total_n = sum(item["n"] for item in series)
    weighted_sum = sum(item["n"] * (item["mid"] - mean_val) ** 2 for item in series)
    return weighted_sum / total_n


def calc_interval_stats(d, series, h):
    m_int = calc_interval_mean(series)
    var_int = calc_interval_variance(series, m_int)
    s_int = math.sqrt(var_int)

    max_idx = 0
    max_n = 0
    for i, item in enumerate(series):
        if item["n"] > max_n:
            max_n = item["n"]
            max_idx = i

    mod_item = series[max_idx]
    a_mod = float(mod_item["intv"].split("-")[0])
    n_prev = series[max_idx - 1]["n"] if max_idx > 0 else 0
    n_next = series[max_idx + 1]["n"] if max_idx < len(series) - 1 else 0

    Mo = a_mod + h * (mod_item["n"] - n_prev) / (
        (mod_item["n"] - n_prev) + (mod_item["n"] - n_next)
    )

    total_n = sum(item["n"] for item in series)
    med_pos = total_n / 2
    med_idx = 0
    s_cum = 0
    for i, item in enumerate(series):
        s_cum += item["n"]
        if s_cum >= med_pos:
            med_idx = i
            break

    med_item = series[med_idx]
    a_med = float(med_item["intv"].split("-")[0])
    s_prev = series[med_idx - 1]["s"] if med_idx > 0 else 0

    Me = a_med + h * (med_pos - s_prev) / med_item["n"]

    r = (min(d) + k * h) - min(d)
    v = (s_int / m_int) * 100 if m_int != 0 else 0

    return {
        "m": m_int,
        "var": var_int,
        "s": s_int,
        "Mo": Mo,
        "n_Mo": max_n,
        "Me": Me,
        "R": r,
        "V": v,
        "min": min(d),
        "max": max(d),
    }


def main():
    fname = "./lab1/lab1_data.txt"
    d = load_file(fname)

    disc_series = make_discrete_series(d)
    int_series, h_val, xmax = make_interval_series(d)

    m = mean(d)
    var = variance(d, m)
    s = std_dev(var)
    Mo, n_Mo = mode_and_freq(d)
    Me = median(d)
    r = range_val(d)
    v = cv(s, m)

    stats_disc = {
        "Средняя": m,
        "Дисперсия": var,
        "СКО": s,
        "Мода": Mo,
        "Частота моды": n_Mo,
        "Медиана": Me,
        "Размах": r,
        "Мин": min(d),
        "Макс": max(d),
        "КВ, %": v,
    }

    stats_int_dict = calc_interval_stats(d, int_series, h_val)
    stats_int = {
        "Средняя": stats_int_dict["m"],
        "Дисперсия": stats_int_dict["var"],
        "СКО": stats_int_dict["s"],
        "Мода": stats_int_dict["Mo"],
        "Частота моды": stats_int_dict["n_Mo"],
        "Медиана": stats_int_dict["Me"],
        "Размах": stats_int_dict["R"],
        "Мин": stats_int_dict["min"],
        "Макс": make_interval_series(d)[2],
        "КВ, %": stats_int_dict["V"],
    }

    print("\nДИСКРЕТНЫЙ РЯД:")
    for k, v in stats_disc.items():
        if k in ["Мода", "Медиана", "Мин", "Макс", "Частота моды"]:
            print(f"{k}: {v:.0f}")
        else:
            print(f"{k}: {v:.3f}")

    print("\nИНТЕРВАЛЬНЫЙ РЯД:")
    for k, v in stats_int.items():
        if k in ["Мин", "Макс", "Частота моды"]:
            print(f"{k}: {v:.0f}")
        elif k in ["Мода", "Медиана"]:
            print(f"{k}: {v:.1f}")
        else:
            print(f"{k}: {v:.3f}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    x_vals = [item["x"] for item in disc_series]
    n_vals = [item["n"] for item in disc_series]
    plt.plot(x_vals, n_vals, "bo-")
    plt.title("Полигон частот (дискретный ряд)")
    plt.xlabel("Возраст")
    plt.ylabel("Частота")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    intvs = [tuple(map(float, item["intv"].split("-"))) for item in int_series]
    widths = [b - a for a, b in intvs]
    lefts = [a for a, b in intvs]
    freqs = [item["n"] for item in int_series]
    plt.bar(lefts, freqs, width=widths, alpha=0.7, edgecolor="black")
    plt.title("Гистограмма (интервальный ряд)")
    plt.xlabel("Возраст")
    plt.ylabel("Частота")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("graphs.png")
    plt.show()


if __name__ == "__main__":
    main()
