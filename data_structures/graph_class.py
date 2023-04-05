class Vertex:
    """
    Класс для хранения информации о вершинах графа.
    """
    def __init__(self, key):
        self.key = key
        self.connections = {}

    def add_adj(self, vertex, weight=0):
        self.connections[vertex] = weight

    def get_key(self):
        return self.key

    def get_connections(self):
        res = ''
        for elem in self.connections:
            res += f'{elem.key} : {self.connections[elem]}, '
        return res


class Graph:
    """
    Класс реализует абстрактную структуру данных — граф.
    Позволяет добавить вершины в граф, установить ребра между ними, а также при необходимости указать вес ребра
    от одной вершины к другой.
    Реализация: в граф можно добавлять только по одной вершине. Аналогично и для установления ребер и веса.
    Чтобы вывести граф со всеми вершинами в классе реализован магический метод __str__.
    """
    def __init__(self):
        self.vertex_dict = {}

    def add_vertex(self, data):
        self.vertex_dict[data] = Vertex(data)

    def add_edges(self, f, t, weight):
        if f not in self.vertex_dict:
            self.add_vertex(f)
        if t not in self.vertex_dict:
            self.add_vertex(t)
        self.vertex_dict[f].add_adj(self.vertex_dict[t], weight)

    def __str__(self):
        res = ''
        for elem in self.vertex_dict:
            res += f'Вершина = {elem}, соседи = {self.vertex_dict[elem].get_connections()}\n'
        return res
