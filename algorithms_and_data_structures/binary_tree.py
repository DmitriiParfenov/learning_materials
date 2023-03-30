class Node:
    """
    Класс для хранения информации об элементе и о ссылке на хранение следующего элемента
    """
    def __init__(self, data):
        self.data = data
        self.left = self.right = None


class BinaryTree:
    """
    Класс для реализации структуры данных, как двоичное дерево
    """
    def __init__(self):
        self.root = None

    def insert(self, n):
        """
        Функция, добавляющая число в двоичное дерево в соответствие с тем, что
        у каждого узла только 2 дочерних узла, причем двоичное дерево хранит узлы в
        отсортированном порядке, где значения каждого узла больше любого значения в
        его левом поддереве и меньше любого в его правом поддереве
        :param n: int
        """
        if not self.root:
            self.root = Node(n)
            return
        elif n < self.root.data:
            if not self.root.left:
                self.root.left = BinaryTree()
            self.root.left.insert(n)
        elif n > self.root.data:
            if not self.root.right:
                self.root.right = BinaryTree()
            self.root.right.insert(n)

    def bfs(self, item):
        """
        Осуществляет поиск элемента в дереве по алгоритму "Поиск в ширину"
        :param item: int
        :return: bool
        """
        used = []
        queue_nodes = []
        queue_nodes += [self.root]
        while queue_nodes:
            node = queue_nodes[0]
            queue_nodes.remove(node)
            if node.data not in used:
                if node.data == item:
                    return True
                elif item < node.data:
                    if not node.left:
                        return False
                    queue_nodes += [node.left.root]
                    used += [node.data]
                elif item > node.data:
                    if not node.right:
                        return False
                    queue_nodes += [node.right.root]
                    used += [node.data]
        return False
