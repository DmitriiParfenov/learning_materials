from abc import ABC, abstractmethod


# Интерфейс для выполнения кнопок.
class Command(ABC):
    """Абстрактный класс для реализации ВЫПОЛНЕНИЯ какой-либо кнопки."""

    @abstractmethod
    def execute(self):
        pass


# Конкретные команды.
class LightOnCommand(Command):
    """Выполнение команды для включения света."""

    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_on()


class LightOffCommand(Command):
    """Выполнение команды для выключения света."""
    def __init__(self, light):
        self.light = light

    def execute(self):
        self.light.turn_off()


# Получатель.
class Light:
    """Класс для включения / выключения света."""
    def turn_on(self):
        print("The light is on")

    def turn_off(self):
        print("The light is off")


# 4. Инициатор
class RemoteControl:
    def __init__(self):
        self.command = None

    def set_command(self, command):
        self.command = command

    def press_button(self):
        self.command.execute()


class MacroCommand:
    """Команда, выполняющая список команд."""

    def __init__(self, commands):
        self.commands = list(commands)

    def __call__(self):
        for command in self.commands:
            command()


if __name__ == "__main__":
    # Определили получателя.
    light = Light()

    # Определили кнопки для получателя.
    light_on = LightOnCommand(light)
    light_off = LightOffCommand(light)

    # Определили инициатора.
    remote = RemoteControl()

    remote.set_command(light_on)
    remote.press_button()  # Вывод: The light is on

    remote.set_command(light_off)
    remote.press_button()  # Вывод: The light is off
