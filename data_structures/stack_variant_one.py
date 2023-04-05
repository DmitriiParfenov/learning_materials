class Stack:
    """
    Класс для реализации стека в виде списка
    """
    __slots__ = ['stack']

    def __init__(self):
        self.stack = []

    def push(self, item):
        """
        Функция, добавляющая объект в самый конец стека
        :param item: object
        """
        self.stack.append(item)

    def pop(self):
        """
        Извлечение объекта из стека по принципу LIFO
        """
        if not len(self.stack):
            raise ValueError('stack is empty')
        self.stack.pop()

    def peek(self):
        """
        Функция возвращает последний элемент из стека
        :return: object
        """
        return self.stack[-1]

    def __str__(self):
        return f'{self.stack}'
