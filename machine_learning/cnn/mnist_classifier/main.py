from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


class MnistPredictor:
    def __init__(self, path_to_model: str) -> None:
        self.path_to_model = Path(path_to_model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def transform_image(self, path_to_image: str, transform=None) -> torch.Tensor:
        img = Image.open(path_to_image)
        img = ImageOps.invert(img)
        if not transform:
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
        return transform(img).unsqueeze(0).to(self.device)  # из [1*28*28] в [1*1*28*28] тензор

    def load_model(self) -> nn.Module:
        model = torch.load(self.path_to_model, weights_only=False)
        model.eval().to(self.device)
        return model

    def predict(self, path_to_image: str, transform=None):
        img_tensor = self.transform_image(path_to_image, transform)
        model = self.load_model()
        y_pred = model(img_tensor)
        return torch.argmax(y_pred, dim=1)

    def predict_proba(self, path_to_image: str, transform=None):
        img_tensor = self.transform_image(path_to_image, transform)
        model = self.load_model()
        y_pred = model(img_tensor)
        return torch.softmax(y_pred, dim=1) * 100
