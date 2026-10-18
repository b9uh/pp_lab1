"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    return sum(1 for char in text.lower() if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(text) == len(set(text))


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        mult = 1
        for digit in str(number):
            mult *= int(digit)
        number = mult
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    result = 0
    for i in range(len(predicted)):
        result += (predicted[i] - expected[i]) ** 2
    return result / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts = []
    divisor = 2
    while divisor * divisor <= number:
        if number % divisor == 0:
            power = 0
            while number % divisor == 0:
                number //= divisor
                power += 1
            parts.append((divisor, power))
        divisor += 1
    if number > 1:
        parts.append((number, 1))

    result = ""
    for base, power in parts:
        if power == 1:
            result += f"({base})"
        else:
            result += f"({base}**{power})"
    return result


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count:
        return k
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(d) for d in str(number)]
    n = len(digits)
    if n % 2 == 1:
        half = n // 2
    else:
        half = n // 2 - 1
    left = sum(digits[:half])
    right = sum(digits[n - half:])
    return left == right
