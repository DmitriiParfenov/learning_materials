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


@pytest.mark.parametrize("item_1, item_2, expected", [(5, 6, 5),
                                                      ('a', 'b', 'a')])
def test_queue_dequeue_1(example_queue, item_1, item_2, expected):
    example_queue.enqueue(item_1)
    example_queue.enqueue(item_2)
    assert example_queue.dequeue() == expected


def test_queue_dequeue_2(example_queue):
    with pytest.raises(ValueError):
        example_queue.dequeue()


def test_queue_str_1(example_queue):
    assert str(example_queue) == ''


@pytest.mark.parametrize("index_outer, expected", [(0, 'data = 6'), (1, 'data = 7')])
def test_queue_str_2(example_queue, index_outer, expected):
    example_queue.enqueue(5)
    example_queue.enqueue(6)
    example_queue.enqueue(7)
    example_queue.dequeue()
    db = str(example_queue).split('\n')
    assert db[index_outer][:8] == expected


def test_queue_get_size_1(example_queue):
    assert example_queue.get_size() == 0


def test_queue_get_size_2(example_queue):
    example_queue.enqueue(5)
    example_queue.enqueue(6)
    assert example_queue.get_size() == 2


def test_queue_get_size_3(example_queue):
    example_queue.enqueue(5)
    example_queue.enqueue(6)
    example_queue.dequeue()
    assert example_queue.get_size() == 1


