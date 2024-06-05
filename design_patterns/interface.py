from abc import ABCMeta, abstractmethod


class IRectangle(metaclass=ABCMeta):
    """Абстрактный класс для реализации методов, присущих всем треугольникам."""

    @property
    @abstractmethod
    def width(self):
        return

    @property
    @abstractmethod
    def height(self):
        return

    @width.setter
    @abstractmethod
    def width(self, value):
        """Set the width of the rectangle."""
        pass

    @height.setter
    @abstractmethod
    def height(self, value):
        """Set the height of the rectangle."""
        pass

    @abstractmethod
    def area(self):
        """Return rectangle area."""
        pass

    @abstractmethod
    def perimeter(self):
        """ Return rectangle perimeter."""
        pass

    @classmethod
    def __subclasshook__(cls, class_child):
        if cls is IRectangle:
            if all([
                any("area" in B.__dict__ for B in class_child.__mro__),
                any("perimeter" in B.__dict__ for B in class_child.__mro__),
                any("width" in B.__dict__ for B in class_child.__mro__),
                any("height" in B.__dict__ for B in class_child.__mro__),
                class_child.__dict__["width"].fset,
                class_child.__dict__["height"].fset,
            ]):
                return True
        raise TypeError("Can't instantiate abstract class Rectangle with abstract method height")


class Rectangle(IRectangle):
    def __init__(self, width, height):
        self._width = width
        self._height = height

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @width.setter
    def width(self, value):
        self._width = value

    @height.setter
    def height(self, value):
        self._height = value

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return self.width * 2 + self.height * 2


if __name__ == "__main__":
    rectangle = Rectangle(10, 10)

    print("Instance type:", type(rectangle))
    print("Instance name:", rectangle.__class__.__name__)
    print("Instance MRO: ", rectangle.__class__.mro())
    isinstance(rectangle, IRectangle)

