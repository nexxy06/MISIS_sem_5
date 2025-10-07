import math


def entropy(p):
    return -p * math.log2(p) if p > 0 else 0


def conditional_entropy_Y_given_Xk(k):
    """Вычисляет H(Y|X) для извлечения k шаров"""
    total_balls = 15
    black = 5
    white = 10

    H_conditional = 0

    # Для каждого возможного количества черных в предварительной выборке
    for i in range(max(0, k - white), min(k, black) + 1):
        # i - черные, (k-i) - белые в предварительной выборке
        prob_config = (math.comb(black, i) * math.comb(white, k - i)) / math.comb(
            total_balls, k
        )

        # После извлечения k шаров осталось:
        black_remaining = black - i
        white_remaining = white - (k - i)
        total_remaining = total_balls - k

        # Вероятности для основного опыта Y
        p_black = black_remaining / total_remaining
        p_white = white_remaining / total_remaining

        H_conditional += prob_config * (entropy(p_black) + entropy(p_white))

    return H_conditional


# Основные вычисления
H_Y = entropy(1 / 3) + entropy(2 / 3)
print(f"H(Y) = {H_Y:.3f}")

for k in [1, 2, 13, 14]:
    H_Y_given_X = conditional_entropy_Y_given_Xk(k)
    I_XY = H_Y - H_Y_given_X
    print(f"K={k}: I(X,Y) = {I_XY:.4f}")
