def depth_first_search(graph, start, end, used=None):
    """
    Рекурсивная функция, которая позволяет обходить все элементы графа (в виде словаря).
    Если искомый элемент есть в графе, то вернет True, если его нет — False
    :param graph: dict
    :param start: object
    :param end: object
    :param used: None
    :return: bool
    """
    if used is None:
        used = []
    used += [start]
    if start == end:
        return True
    elif start not in graph:
        return False
    for elem in graph[start]:
        if elem not in used:
            dfs = depth_first_search(graph, elem, end, used)
            if dfs:
                return dfs
    return False