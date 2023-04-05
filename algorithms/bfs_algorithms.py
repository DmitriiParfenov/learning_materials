from collections import deque


def breadth_first_search(graph, start, target):
    """
    Функция реазилует алгоритм "Поиск в ширину", который позволяет обходить все элементы графа и найти
    путь с минимальным количество сегментов
    :param graph: dict
    :param start: hashable object
    :param target: object
    :return: bool
    """
    queue_func = deque()
    queue_func += graph[start]
    used = []
    while queue_func:
        node = queue_func.popleft()
        if node not in used:
            if node == target:
                return True
            else:
                queue_func += graph[node]
                used.append(node)
    return False

