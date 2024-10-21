"""
Игра "Угадай число".
Компьютер сам загадывает и сам угадывает число.
"""

import numpy as np


def game_core_v3(number: int) -> int:
    """
    Компьютер угадывает загаданное число, используя динамическое обновление диапазона.
    Функция принимает загаданное число и возвращает количество попыток.
       
    Args:
        number (int): Загаданное число.

    Returns:
        int: Число попыток.
    """
    count = 0
    a = 1
    b = 101 # переменные, которые отвечают за диапазон угадывания
    predict = np.random.randint(a, b)  # генерируем начальное предполагаемое число в заданном диапазоне 1 - 100

    while number != predict:
        count += 1
        print(f"Шаг {count}: Загаданное число: {number}, Предполагаемое число: {predict}, Диапазон: ({a}, {b})")
        if predict > number:
            b = predict  # уменьшаем верхнюю границу
        elif predict < number:
            a = predict + 1  # увеличиваем нижнюю границу
        predict = np.random.randint(a, b)  # генерируем новое предполагаемое число после коррректировки диапазона

    count += 1
    print(f"Шаг {count}: Загаданное число: {number}, Предполагаемое число: {predict}, Диапазон: ({a}, {b})")
    return count

def score_game(game_core_v3) -> int:
    """За какое количство попыток в среднем за 1000 подходов угадывает наш алгоритм

    Args:
        game_core_v3([type]): функция угадывания

    Returns:
        int: среднее количество попыток
    """
    count_ls = []
    #np.random.seed(1)  # фиксируем сид для воспроизводимости
    random_array = np.random.randint(1, 101, size=(1000))  # загадали список чисел

    for number in random_array:
        count_ls.append(game_core_v3(number))

    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за:{score} попыток")
    return score


if __name__ == "__main__":
    # RUN
    score_game(game_core_v3)
