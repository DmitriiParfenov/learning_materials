class Event:
    """Класс Event, реализующий паттерн Наблюдатель."""

    _observers = []

    def __init__(self, subject):
        """Экземпляр инициализирует объект subject."""
        self.subject = subject

    @classmethod
    def register(cls, observer):
        """Метод добавляет в _observers объект observer, если его нет в _observers."""
        if observer not in cls._observers:
            cls._observers.append(observer)

    @classmethod
    def unregister(cls, observer):
        """Метод удаляет из _observers объект observer, если он есть в _observers."""
        if observer in cls._observers:
            cls._observers.remove(observer)

    @classmethod
    def notify(cls, subject):
        """Метод реализует рассылку уведомлений всем зарегистрированным наблюдателям (observers) о
        каком-либо событии (экземпляр класса Event)."""
        event = cls(subject)  # создали экземпляр класса Event.
        for observer in cls._observers:
            observer(event)


class WriteEvent(Event):
    """Класс для реализации события для примера."""
    def __repr__(self):
        return 'WriteEvent'


def log(event):
    print(f'Произошло событие {event} с объектом {event.subject}.')


class AnotherObserver:
    def __call__(self, event):
        print(f"Событие {event} запустило действие {self.__class__.__name__}.")


# В событии для примера зарегистрировали наблюдателей.
WriteEvent.register(log)
WriteEvent.register(AnotherObserver())

if __name__ == "__main__":
    # Так как в Event есть наблюдатели, то вызов метода оповестит их об этом.
    Event.notify("telephone rang")
