class Node:
    """
    Класс для хранения информации об узле и о ссылке на хранение следующего узла
    """
    def __init__(self, data, next_node=None):
        self.data = data
        self.next_node = next_node


class LinkedList:
    """
    Класс для создания связного списка с методами
    """
    def __init__(self):
        self.head = None

    def append(self, item):
        """
        Добавление элемента в конец списка
        """
        if not self.head:
            self.head = Node(item)
            return
        current = self.head
        while current.next_node:
            current = current.next_node
        current.next_node = Node(item)

    def remove(self, item):
        """
        Удаление элемента из связного списка
        """
        if self.head.data == item:
            self.head = self.head.next_node
            return
        current = self.head
        previous = None
        while current:
            if current.data == item:
                previous.next_node = current.next_node
            previous = current
            current = current.next_node

    def search(self, item):
        """
        Поиск элемента в связном списке
        :param item: object
        :return: bool
        """
        current = self.head
        while current:
            if current.data == item:
                return True
            else:
                current = current.next_node
        return False

    def reversed_list(self):
        """
        Сортирует узлы в связном списке в обратном порядке
        """
        current = self.head
        previous = None
        while current:
            next_node = current.next_node
            current.next_node = previous
            previous = current
            current = next_node
        self.head = previous

    def __str__(self):
        res = ''
        node = self.head
        while node:
            res += f'Узел = {node.data}, ссылка = {node.next_node}\n'
            node = node.next_node
        return res
