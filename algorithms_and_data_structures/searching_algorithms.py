def binary_search(arr, target):
    """
    Осуществляет поиск таргетного элемента в массиве
    при помощи бинарного поиска. O(logn)
    :param arr: list
    :param target: int
    :return: int
    """
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        guess = arr[mid]
        if guess == target:
            return mid
        elif guess > target:
            high = mid - 1
        else:
            low = mid + 1
    return False
