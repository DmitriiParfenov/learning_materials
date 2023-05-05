import pytest

from data_structures.linked_list import LinkedList


@pytest.fixture
def linked_list_example():
    return LinkedList()


def test_empty_linked_list(linked_list_example):
    assert linked_list_example.head is None


def test_linked_list_append_1(linked_list_example):
    linked_list_example.append(5)
    data = [linked_list_example.head.data, linked_list_example.head.next_node]
    assert data == [5, None]


def test_linked_list_append_2(linked_list_example):
    linked_list_example.append(5)
    linked_list_example.append(6)
    linked_list_example.append(7)
    linked_list_example.append(9)
    data = list()
    while linked_list_example.head:
        data.append(linked_list_example.head.data)
        linked_list_example.head = linked_list_example.head.next_node
    assert data == [5, 6, 7, 9]


def test_linked_list_remove_1(linked_list_example):
    linked_list_example.append(5)
    linked_list_example.remove(5)
    assert linked_list_example.head is None


def test_linked_list_remove_2(linked_list_example):
    linked_list_example.append(5)
    linked_list_example.remove(6)
    assert linked_list_example.head.data == 5


@pytest.mark.parametrize("item, expected", [(5, True), (7, True), (9, True), (-5, False), (0, False), (10, False)])
def test_linked_list_search(linked_list_example, item, expected):
    linked_list_example.append(5)
    linked_list_example.append(7)
    linked_list_example.append(9)
    assert linked_list_example.search(item) == expected


def test_linked_list_reversed(linked_list_example):
    linked_list_example.append(2)
    linked_list_example.append(4)
    linked_list_example.append(6)
    linked_list_example.reversed_list()
    data = list()
    while linked_list_example.head:
        data.append(linked_list_example.head.data)
        linked_list_example.head = linked_list_example.head.next_node
    assert data == [6, 4, 2]


def test_linked_list_str(linked_list_example):
    linked_list_example.append(2)
    linked_list_example.append(4)
    linked_list_example.append(6)
    data = str(linked_list_example).split('\n')[:-1]
    result = list()
    for elem in data:
        result.append(elem[:8])
    assert result == ['Узел = 2', 'Узел = 4', 'Узел = 6']
