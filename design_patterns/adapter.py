from os.path import split, splitext


class DublinCoreAdapter:
    """Класс адаптер для работы с файлами. Экземпляр класса инициализируется полем - название файла."""

    def __init__(self, filename) -> None:
        self._filename = filename

    @property
    def title(self) -> str:
        """Возвращает имя файла без расширения. Извлекает последнюю часть пути к файлу, игнорируя директории, и удаляет
        расширение файла, оставляя только его базовое имя. """
        return splitext(split(self._filename)[-1])[0]

    @property
    def languages(self) -> tuple:
        """ Возвращает кортеж поддерживаемых языков. Этот метод предоставляет список языков, которые поддерживаются в
        данном контексте. В текущей реализации возвращается только английский язык.
        """
        return 'en',

    def __getattr__(self, item) -> str:
        """Магический метод, который возвращает <Unknown> при обращении к несуществующему атрибуту класса."""
        return 'Unknown'


class DublinCoreInfo:
    """Класс, который отображает информацию об объектах DublinCoreInfo (реализация опущена). Чтобы иметь возможность
    отображать информацию из файлов, необходимо адаптировать текущий класс для работы с файлами, который имеет поля
    (необязательно) - title, languages, creator."""

    @staticmethod
    def summary(adapter: DublinCoreAdapter) -> str:
        """Метод позволяет получить информацию из файлов через адаптер. """
        result = f'Title: {adapter.title}\nLanguage: {", ".join(adapter.languages)}\nCreator: {adapter.creator}'
        return result


if __name__ == "__main__":
    file_name = 'example.txt'
    summary_obj = DublinCoreInfo()
    adapter_obj = DublinCoreAdapter(file_name)
    print(f'Summary of {file_name} adapted with DublinCoreAdapter')
    print(summary_obj.summary(adapter_obj))

