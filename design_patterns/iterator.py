import re
import reprlib

RE_WORD = re.compile(r'\w+')


# ПЕРВЫЙ ВАРИАНТ РЕАЛИЗАЦИИ ПАТТЕРНА.

# Класс Sentence является итерируемым объектом, так как реализует только метод __iter__, который возвращает объект
# SentenceIterator. SentenceIterator - это объект итератор, так как реализует __iter__ и __next__. Класс
# SentenceIterator необходим, чтобы мы могли обращаться за новым независимым итератором. Иными словами, у нас может быть
# много итераторов со своим состоянием (какой-то итератор сейчас на первой позиции, другой - на другой и тд), тем самым
# мы можем проходиться по итерируемому объекту Sentence несколько раз ПАРАЛЛЕЛЬНО, так как у нас много НЕЗАВИСИМЫХ
# итераторов SentenceIterator.
class SentenceFirst:
    def __init__(self, text: str):
        self.text = text
        self.words = RE_WORD.findall(text)

    def __repr__(self):
        result = f'{reprlib.repr(self.text)}'
        return result

    def __iter__(self):
        return SentenceIterator(self.words)


class SentenceIterator:
    def __init__(self, words):
        self.words = words
        self.index = 0

    def __next__(self):
        try:
            word = self.words[self.index]
        except IndexError:
            raise StopIteration()
        self.index += 1
        return word

    def __iter__(self):
        return self


# ВТОРОЙ ВАРИАНТ РЕАЛИЗАЦИИ ПАТТЕРНА.

# Использование объектов генераторов, так как они являются итераторами.
class SentenceSecond:
    def __init__(self, text):
        self.text = text
        self.words = RE_WORD.findall(text)

    def __repr__(self):
        result = f'{reprlib.repr(self.text)}'
        return result

    def __iter__(self):
        for word in self.words:
            yield word


# ТРЕТИЙ ВАРИАНТ РЕАЛИЗАЦИИ ПАТТЕРНА.

# Ленивое вычисление, так как все найденный слова мы не записываем в переменную self.words, а храним в генераторе,
# который порождает объекты re.MatchObject по запросу - экономия памяти.
class SentenceThird:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        result = f'{reprlib.repr(self.text)}'
        return result

    def __iter__(self):
        for match in RE_WORD.finditer(self.text):
            yield match.group()

    # Аналогично.
    # def __iter__(self):
    #     return (match.group() for match in RE_WORD.finditer(self.text))
