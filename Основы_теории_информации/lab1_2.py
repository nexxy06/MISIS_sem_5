import math


def entropy(p1, p2):
    """Энтропия для двух событий"""
    result = 0
    if p1 > 0:
        result += -p1 * math.log2(p1)
    if p2 > 0:
        result += -p2 * math.log2(p2)
    return result


def solve_urn_problem(n, m):
    """Решает задачу для урны с n шарами, m чёрных"""

    print(f"Урна: {m} чёрных + {n-m} белых = {n} шаров")

    # 1. Энтропия H(X)
    p_x1 = m / n
    p_x2 = (n - m) / n
    H_X = entropy(p_x1, p_x2)
    print(f"\n1. Энтропия H(X) = {H_X:.4f}")

    # 2. Энтропия H(Y)
    H_Y = H_X  # симметрия
    print(f"2. Энтропия H(Y) = {H_Y:.4f}")

    # 3. Условная энтропия H_X(Y)
    if n > 1:
        # Случай 1: первый шар чёрный
        if m > 1:
            H_Y_given_x1 = entropy((m - 1) / (n - 1), (n - m) / (n - 1))
        else:
            H_Y_given_x1 = entropy(0, 1)  # только белые остались

        # Случай 2: первый шар белый
        if n - m > 1:
            H_Y_given_x2 = entropy(m / (n - 1), (n - m - 1) / (n - 1))
        else:
            H_Y_given_x2 = entropy(1, 0)  # только чёрные остались

        H_X_Y = p_x1 * H_Y_given_x1 + p_x2 * H_Y_given_x2
        print(f"3. Условная энтропия H_X(Y) = {H_X_Y:.4f}")

        # 4. Условная энтропия H_Y(X)
        H_Y_X = H_X_Y  # симметрия
        print(f"4. Условная энтропия H_Y(X) = {H_Y_X:.4f}")

        # 5. Информация взаимная
        I_XY = H_X - H_Y_X
        print(f"5. Взаимная информация I(X,Y) = {I_XY:.4f}")

    return H_X, H_Y, H_X_Y


# Пример вычисления
print("=== РЕШЕНИЕ ЗАДАЧИ ===")
H_X, H_Y, H_XY = solve_urn_problem(n=10, m=4)
