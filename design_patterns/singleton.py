class SingletonNewMethod:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f'Name {self.name}, age {self.age}.'


class SingletonMeta(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__call__(*args, **kwargs)

        return cls._instance


class Student(metaclass=SingletonMeta):

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f'Name {self.name}, age {self.age}.'


def singleton_decorator(aClass):
    instance = None

    def wrapper(*args, **kwargs):
        nonlocal instance
        if not instance:
            instance = aClass(*args, **kwargs)
        return instance

    return wrapper


@singleton_decorator
class StudentDecorator:

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __str__(self):
        return f'Name {self.name}, age {self.age}.'
