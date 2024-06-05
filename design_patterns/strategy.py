from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import Decimal
from typing import NamedTuple, Optional


class Customer(NamedTuple):
    """
    Класс для идентификации покупателя, где name - это имя покупателя, а fidelity - это баллы лояльности покупателя.
    """
    name: str
    fidelity: int


class LineItem(NamedTuple):
    """
    Класс для представления товара в заказе, где product - название продукта, quantity - количество продукта, а price -
    это цена за единицу продукта.
    """
    product: str
    quantity: int
    price: Decimal

    def total(self) -> Decimal:
        """Метод возвращает общую стоимость товара."""
        return self.price * self.quantity


class Order(NamedTuple):
    """
    Класс для создания заказа, где customer - это покупатель, cart - товары для покупки, а promotion - это расчет скидки
    в зависимости от характеристик покупателя (его баллов лояльности, количества товара и тд) - иными словами в
    способ расчета будет изменяться независимо от желания самого клиента, а только исходя из его характеристик.
    """
    customer: Customer
    cart: Sequence[LineItem]
    promotion: Optional['Promotion'] = None

    def total(self) -> Decimal:
        """Расчет общей стоимости заказа в зависимости от количества товаров в cart."""
        totals = (item.total() for item in self.cart)
        return sum(totals, start=Decimal(0))

    def due(self) -> Decimal:
        """Расчет скидки в зависимости от характеристик покупателя - тут реализуется паттерн Стратегия."""
        if self.promotion is None:
            discount = Decimal(0)
        else:
            discount = self.promotion.discount(self)
        return self.total() - discount

    def __repr__(self):
        return f'Общая стоимость заказа: {self.total():.2f}, со скидкой: {self.due():.2f}.'


class Promotion(ABC):
    @abstractmethod
    def discount(self, order: Order) -> Decimal:
        """Вернуть скидку в виде положительной суммы в долларах"""


class FidelityPromo(Promotion):  # первая конкретная стратегия
    """5%-ная скидка для заказчиков, имеющих не менее 1000 баллов лояльности"""

    def discount(self, order: Order) -> Decimal:
        rate = Decimal('0.05')
        if order.customer.fidelity >= 1000:
            return order.total() * rate
        return Decimal(0)


class BulkItemPromo(Promotion):  # вторая конкретная стратегия
    """10%-ная скидка для каждой позиции LineItem, в которой заказано не менее 20 единиц"""

    def discount(self, order: Order) -> Decimal:
        discount = Decimal(0)
        for item in order.cart:
            if item.quantity >= 20:
                discount += item.total() * Decimal('0.1')
        return discount


class LargeOrderPromo(Promotion):  # третья конкретная стратегия
    """7%-ная скидка для заказов, включающих не менее 10 различных позиций"""

    def discount(self, order: Order) -> Decimal:
        distinct_items = {item.product for item in order.cart}
        if len(distinct_items) >= 10:
            return order.total() * Decimal('0.07')
        return Decimal(0)
