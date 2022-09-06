from pathlib import Path
from torchvision.transforms import ToTensor, ToPILImage
import torch
from PIL import Image
from utils.config_parser import ConfigParser


def do_multiple_diffusion_steps(img, alpha_head):
    e = torch.normal(0, 1, img.shape)
    return torch.sqrt(alpha_head) * img + torch.sqrt(1 - alpha_head) * e


if __name__ == '__main__':
    image_path = "~/Documents/data/aml/original128png/00018.png"
    image_path = Path(image_path).expanduser()

    config_file = "../configs/diffusion_model.yaml"
    config = ConfigParser.read(config_file)

    diffusion_betas = torch.linspace(
            config["training"]["diffusion_beta_1"],
            config["training"]["diffusion_beta_capital_t"],
            config["training"]["diffusion_steps"]
            )
    img_orig = ToTensor()(Image.open(image_path))[None, :] * 2 - 1

    for t in [1, 150, 149, 299, 298]:
        alpha_head = torch.prod(1 - diffusion_betas[:t + 1])
        noisy_img = do_multiple_diffusion_steps(img_orig, alpha_head)
        ToPILImage()(torch.clip(noisy_img[0], -1, 1) / 2 + 0.5).save(f"noisy_version_{t}.png")