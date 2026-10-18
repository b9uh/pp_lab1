from grader_contracts.python_basics import TextInput, VectorPairInput
from python_basics import *


def test_count_vowels():
    assert count_vowels(TextInput(value="hello")) == 2
    assert count_vowels(TextInput(value="HELLO")) == 2
    assert count_vowels(TextInput(value="bcd")) == 0
    assert count_vowels(TextInput(value="")) == 0

def test_has_unique_characters():
    assert has_unique_characters(TextInput(value="abcde")) is True
    assert has_unique_characters(TextInput(value="abcba")) is False
    assert has_unique_characters(TextInput(value="")) is True

def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(value=1)) == 1
    assert count_one_bits(PositiveIntegerInput(value=7)) == 3
    assert count_one_bits(PositiveIntegerInput(value=8)) == 1
    assert count_one_bits(PositiveIntegerInput(value=255)) == 8

def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(value=39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(value=4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(value=999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(value=10)) == 1

def test_mse():
    assert mse(VectorPairInput(predicted=[1, 2, 3], expected=[1, 2, 3])) == 0.0
    assert mse(VectorPairInput(predicted=[1, 1], expected=[0, 0])) == 1.0
    assert mse(VectorPairInput(predicted=[2, 4], expected=[1, 1])) == 5.0 

def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(value=86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(value=13)) == "(13)"
    assert prime_factorization(PositiveIntegerInput(value=8)) == "(2**3)"

def test_pyramid():
    assert pyramid(PositiveIntegerInput(value=1)) == 1
    assert pyramid(PositiveIntegerInput(value=5)) == 2
    assert pyramid(PositiveIntegerInput(value=14)) == 3
    assert pyramid(PositiveIntegerInput(value=4)) == "It is impossible"

def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(value=1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(value=123456)) is False
    assert is_balanced_number(PositiveIntegerInput(value=11)) is True


if __name__ == "__main__":
    test_count_vowels()
    test_has_unique_characters()
    test_count_one_bits()
    test_multiplicative_persistence()
    test_mse()
    test_prime_factorization()
    test_pyramid()
    test_is_balanced_number()
    print("OK")