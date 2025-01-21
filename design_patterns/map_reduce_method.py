from functools import reduce

# Для примера каждый элемент списка data - это отдельный поднабор.
data = [
    "I know what I know.",
    "I know that I know.",
    "I don't know that much.",
    "They don't know much.",
]


def get_frequencies(text: str) -> dict[str, int]:
    """
    Метод считает количество слов в text и возвращает словарь, где ключи - это встречаемое слово в text, а значение -
    это частота встречаемости соответствующего слова в строке.
    """
    text_list = text.rstrip('.').split(' ')
    frequencies = {}
    for word in text_list:
        frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


def merge_dict(first: dict[str, int], second: dict[str, int]) -> dict[str, int]:
    """
    Метод объединяет два словаря в один. Если ключи есть в обоих словарях, то их значения просто суммируются.
    """
    result = first
    for word in second:
        result[word] = result.get(word, 0) + second.get(word)
    return result


# Выполнили get_frequencies для каждого поднабора в большом наборе data.
mapping = [get_frequencies(x) for x in data]
# Объединили решения для каждого поднабора в окончательный ответ.
reducing = reduce(merge_dict, mapping)
print(reducing)
