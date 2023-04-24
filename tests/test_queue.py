import pytest

from data_structures.queue import Queue


@pytest.fixture
def example_queue():
    return Queue()


def test_empty_queue(example_queue):
    attrs = []
    for elem in example_queue.__dict__:
        attrs.append(example_queue.__dict__[elem])
    assert attrs == [None, None, 0]


def test_queue_enqueue_size(example_queue):
    example_queue.enqueue(5)
    assert example_queue._size == 1


def test_queue_enqueue_front(example_queue):
    example_queue.enqueue(5)
    assert example_queue.front.data == 5


def test_queue_enqueue_rear(example_queue):
    example_queue.enqueue(5)
    assert example_queue.front.data == 5


def test_queue_enqueue(example_queue):
    example_queue.enqueue(5)
    example_queue.enqueue(6)
    attrs = list()
    attrs.append(example_queue.front.data)
    attrs.append(example_queue.rear.data)
    attrs.append(example_queue._size)
    assert attrs == [5, 6, 2]
