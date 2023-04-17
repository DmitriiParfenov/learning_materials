import pytest

from data_structures.stack_variant_two import Stack


@pytest.fixture
def stack_example():
    return Stack()


def test_stack_empty_stack(stack_example):
    assert stack_example.head is None


def test_stack_push_1(stack_example):
    stack_example.push(5)
    assert stack_example.head.__class__.__name__ == 'Node'


def test_stack_push_1_print(stack_example):
    stack_example.push(5)
    assert stack_example.__str__() == 'Узел — 5 => следующий узел — None\n'


def test_stack_push_2_elements(stack_example):
    stack_example.push(5)
    stack_example.push(6)
    result = []
    while stack_example.head:
        result.append(stack_example.head.data)
        stack_example.head = stack_example.head.next_node
    assert result == [6, 5]


def test_stack_pop_from_empty(stack_example):
    with pytest.raises(ValueError):
        stack_example.pop()


def test_stack_pop(stack_example):
    stack_example.push(5)
    stack_example.push(6)
    result = []
    for i in range(2):
        result.append(stack_example.pop())
    assert result == [6, 5]
