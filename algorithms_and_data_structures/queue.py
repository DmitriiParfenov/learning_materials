class Node:
    """
    Класс для хранения информации об элементе и о ссылке на хранение следующего элемента
    """
    def __init__(self, data, next_node=None):
        self.data = data
        self.next_node = next_node


class Queue:
    def __init__(self):
        self.front = None
        self.rear = None
        self._size = 0

    def enqueue(self, item):
        """
        Функция добавляет элемент в очередь по принципу FIFO
        :param item: object
        """
        self._size += 1
        if not self.rear:
            self.front = self.rear = Node(item)
        else:
            self.rear.next_node = self.rear = Node(item)

    def dequeue(self):
        """
        Функция удаляет элемент из очереди по принципу FIFO
        """
        self._size -= 1
        if not self.front:
            raise ValueError('queue is empty')
        temp = self.front
        self.front = self.front.next_node
        if not self.front:
            self.rear = None
        return temp.data

    def get_size(self):
        """
        Функция возвращает размер очереди
        :return: int
        """
        return self._size

    def __str__(self):
        res = ''
        node = self.front
        while node:
            res += f'data = {node.data}, next_node =  {node.next_node}\n'
            node = node.next_node
        return res
