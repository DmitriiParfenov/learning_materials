import abc


class Validated(abc.ABC):
    """
    Класс-дескриптор, который при установлении атрибута экземпляру клиентского класса делегирует валидацию
    установленного имени.
    """

    def __set_name__(self, owner, name):
        """
        Метод, вызываемый при создании дескриптора, чтобы сохранить имя атрибута, к которому он привязан. Storage_name -
        это атрибут, который будет иметь название name.
        Args:
         - owner (type): Класс, в котором используется дескриптор.
         - name (str): Имя атрибута, к которому привязан дескриптор.
        """
        self.storage_name = name

    def __set__(self, instance, value):
        """
        Метод делегирует валидацию значения указанного атрибута методу validate. Валидные данные записываются в __dict__
        клиентского экземпляра класса.
        """
        value = self.validate(self.storage_name, value)
        instance.__dict__[self.storage_name] = value

    @abc.abstractmethod
    def validate(self, name, value):
        """Вернуть проверенное значение или возбудить ValueError."""


class Quantity(Validated):
    """Подкласс для валидации значения quantity, которое должно быть больше 0."""

    def validate(self, name, value):
        if value <= 0:
            raise ValueError(f'{name} must be > 0')
        return value


class LineItem:
    weight = Quantity()  # Дескриптор.
    price = Quantity()  # Дескриптор.

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price


if __name__ == '__main__':
    item = LineItem('apple', 1.1, 4.99)
    print(item.__dict__)
    item.price = 3.00
    print(item.__dict__)
    item.price = -5  # ValueError
