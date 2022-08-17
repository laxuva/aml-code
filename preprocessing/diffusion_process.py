from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def do_multiple_diffusion_steps(image_0, betas):
    alpha_t = np.prod(1 - np.array(betas))
    return np.random.normal(alpha_t * image_0, (1 - alpha_t), image_0.shape)


if __name__ == '__main__':
    img = (np.array(Image.open(Path("~/Documents/data/aml/original128png/00011.png").expanduser())) / 255 - 0.5) * 2
    betas = np.linspace(0.0001, 0.02, 500)

    plt.imshow(((img / 2 + 0.5) * 255).astype(np.uint8))
    plt.show()

    noised_img = do_multiple_diffusion_steps(img, betas)

    plt.imshow(((np.clip(noised_img, -1, 1) / 2 + 0.5) * 255).astype(np.uint8))
    plt.show()
