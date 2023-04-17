class Node:
    """Класс для хранения информации об узле и о ссылке на хранение следующего узла"""

    def __init__(self, data, next_node=None):
        self.data = data
        self.next_node = next_node


class Stack:
    """Класс для реализации стэка в виде связного списка"""

    def __init__(self):
        self.head = None

    def push(self, item):
        """
        Функция добавляет элемент в конец стэка
        :param item: object
        :return: None
        """
        node = Node(item)
        if not self.head:
            self.head = node
        else:
            node.next_node = self.head
            self.head = node

    def pop(self):
        """
        Функция удаляет из стэка самый последний элемент
        """
        if not self.head:
            raise ValueError('pop from empty stack')
        current = self.head
        self.head = self.head.next_node
        return current.data

    def __str__(self):
        res = ''
        node = self.head
        while node:
            res += f'Узел — {node.data} => следующий узел — {node.next_node}\n'
            node = node.next_node
        return res

