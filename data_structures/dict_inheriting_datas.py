def get_dict_inhere(structure, parent, child):
    """
    Функция реализует структуру данных, которая отражает наследование объектов.
    Structure — это словарь, ключами которого являются объекты, а значения этох ключей  — это
    список объектов, унаследованных от объекта ключа
    :param structure: dict
    :param parent: hashable object
    :param child: list
    :return: dict
    """
    if parent not in structure:
        structure[parent] = []
        structure[parent].extend(child)
        for elem in structure[parent]:
            if elem in structure:
                structure[parent].extend(structure[elem])
        structure[parent] = set(structure[parent])
    return structure
