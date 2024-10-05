from collections import deque


def breadth_first_search(graph, start, target):
    """
    Функция реализует алгоритм "Поиск в ширину", который позволяет обходить все элементы графа и определить, наследуется
    ли узел start от узла target. Обход графа совершается поуровнево.
    :param graph: dict
    :param start: hashable object
    :param target: object
    :return: bool
    """
    if start not in graph:
        return False
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

