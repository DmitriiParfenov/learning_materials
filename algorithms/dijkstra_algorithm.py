# Объявление глобальных переменных
used = []
graph = {}
node_costs = {}


def get_graph(parent, child, weight, structure):
    """
    Функция возвращает словарь, где ключами являются родительские узлы, значения ключей — дочерние узлы с весами
    в виде словаря (ключ — узел : значение — вес). Если родительские и дочерние узлы равны, то возбудится исключение
    ValueError.
    :param parent: hashable object
    :param child: object
    :param weight: int
    :param structure: dict
    :return: dict
    """
    if child == parent:
        raise ValueError('child не может наследоваться сам от себя')
    if parent not in graph:
        structure[parent] = {}
        structure[parent][child] = weight
    structure[parent][child] = weight
    structure['end'] = {}
    return structure


def get_weight(structure, costs):
    """
    Функция принимает граф, который должен содержать ключ "start", и словарь, в который добавляются узлы в качестве
    ключей, а значениями являются минимально возможные пути до этих узлов.
    :param structure: dict
    :param costs: dict
    :return: dict
    """
    for key in structure:
        if key == 'start':
            new_keys = list(structure['start'].keys())
            new_values = list(structure['start'].values())
            temp_dict = dict(zip(new_keys, new_values))
            costs.update(temp_dict)
        elif key not in costs:
            costs[key] = float('inf')
    return costs


def find_lowest(weights):
    """
    Функция возвращает узел с самым наименьшим весом
    :param weights: dict
    :return: hashable object
    """
    lowest_cost = float('inf')
    lowest_node = None
    for elem in weights:
        if elem not in used:
            if weights[elem] < lowest_cost:
                lowest_node = elem
                lowest_cost = weights[elem]
    return lowest_node


def main():
    greeting = """Граф должен содержать узел start, он же корневой узел, к которому добавляются дочерние узлы.
Дочерние узлы, родительский узел которых является start, должны родительскими узлами для узла end.
Узел end уже присутствует в графе"""
    print(greeting)
    user_answer = input('Вы хотите увидеть пример графа? [yes/no] ')
    if user_answer.lower() not in ['no', 'n', 'нет', 'н']:
        print({'start': {'a': 6, 'b': 2}, 'a': {'end': 1}, 'b': {'a': 3, 'end': 5}})
    flag = True
    while flag:
        user_input = input('Введите данные в формате — родительский узел, дочерний узел, вес: ')
        try:
            user_parent, user_child, user_weight = user_input.split(', ')
            get_graph(user_parent, user_child, int(user_weight), graph)
            user_input_answer = input('Вы хотите добавить еще данных в граф? [yes/no] ')
            if user_input_answer.lower() == 'no':
                flag = False
        except ValueError:
            print('Вы должны ввести три значения')

    get_weight(graph, node_costs)
    node = find_lowest(node_costs)

    while node:
        cost = node_costs[node]
        neighbors = graph[node]
        for n in neighbors:
            new_cost = cost + neighbors[n]
            if new_cost < node_costs[n]:
                node_costs[n] = new_cost
        used.append(node)
        node = find_lowest(node_costs)


main()
print(node_costs)