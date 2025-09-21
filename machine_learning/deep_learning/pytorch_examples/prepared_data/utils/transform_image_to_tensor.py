import torchvision.transforms as transforms

def decode_image_to_tensor(height: int, width: int) -> transforms.Compose:
    """
    Метод возвращает callable объект, который принимает изображения с последующим преобразованием в тензор заданного
    размера.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((height, width))
    ])