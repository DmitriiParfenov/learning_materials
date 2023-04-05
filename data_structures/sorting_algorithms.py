import random


def quick_sort(arr):
    """
    Осуществляет сортировку массива по принципу "Разделяй и
    властвуй". O(nlogn)
    :param arr: list
    :return: list
    """
    if len(arr) < 2:
        return arr
    else:
        pivot = arr[random.randint(0, len(arr)-1)]
        less = [i for i in arr if i < pivot]
        greater = [i for i in arr if i > pivot]
        return quick_sort(less) + [pivot] + quick_sort(greater)


def bubble_sort(arr):
    """
    Осуществляет сортировку массива методом
    пузырькой сортировки. O(n^2)
    :param arr:
    :return:
    """
    for i in range(len(arr)-1):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr


def union(arr1, arr2):
    """
    Соединяет два массива в один отсортированный массив.
    :param arr1: list
    :param arr2: list
    :return: list
    """
    result = []
    i = j = 0
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            result.append(arr1[i])
            i += 1
        else:
            result.append(arr2[j])
            j += 1
    if i < len(arr1):
        result.extend(arr1[i:])
    if j < len(arr2):
        result.extend(arr2[j:])
    return result


def merge_sort(arr):
    """
    Осуществляет сортировку массива по принципу
    "Разделяй и властвуй". O(nlogn)
    :param arr: list
    :return: list
    """
    if len(arr) < 2:
        return arr
    else:
        mid = len(arr) // 2
        left = quick_sort(arr[:mid])
        right = quick_sort(arr[mid:])
        return union(left, right)
