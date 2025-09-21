import pathlib


def get_file_list(pathway: str, pattern: str) -> list[str]:
    """
    Метод возвращает список путей до файлов с расширением pattern в указанной директории pathway.
    Args:
        pathway: путь до файлов
        pattern: расширение требуемых файлов
    Returns:
        list: список путей до искомых файлов
    """
    images_dir = pathlib.Path(pathway)
    if not images_dir.exists() and not images_dir.is_dir():
        raise FileNotFoundError('Files not found.')
    return [str(path) for path in sorted(images_dir.glob(pattern))]
