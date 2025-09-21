import os

import matplotlib.pyplot as plt
from PIL import Image


def visualize_images(files: list[str], size: tuple[int, int]) -> None:
    fig = plt.figure(figsize=size)
    for idx, path in enumerate(files):
        img = Image.open(path)
        ax = fig.add_subplot(2, 3, idx + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(os.path.basename(path), size=15)
        ax.imshow(img)

    plt.show()
