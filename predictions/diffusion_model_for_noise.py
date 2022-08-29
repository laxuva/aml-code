from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from network.unet_with_embedding import UNet
from utils.config_parser import ConfigParser


@torch.no_grad()
def test_prediction(model_path, image_path, out_path, config_file="../configs/diffusion_model.yaml"):
    config = ConfigParser.read(config_file)

    image_path = Path(image_path).expanduser()
    out_path = Path(out_path).expanduser()

    if not out_path.exists():
        out_path.mkdir()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    diffusion_betas = torch.linspace(
        config["training"]["diffusion_beta_1"],
        config["training"]["diffusion_beta_capital_t"],
        config["training"]["diffusion_steps"]
    ).to(device)

    T = config["training"]["diffusion_steps"]

    model = UNet(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()

    x = ToTensor()(Image.open(image_path)).to(device)[None, :]

    img_shape = list(x.shape)
    img = torch.randn(img_shape, device=device)

    alpha_head_t_minus_one = 0

    for t in tqdm(range(0, len(diffusion_betas))[::-1]):
        alpha_head = torch.prod(1 - diffusion_betas[:t+1]).to(device)
        alpha = 1 - diffusion_betas[t].to(device)

        noise_to_reduce = model.forward(img, torch.tensor([t]).to(device))
        new_value = 1 / torch.sqrt(alpha) * (img - (1 - alpha) / torch.sqrt(1 - alpha_head) * noise_to_reduce)

        if t > 0:
            z = torch.randn_like(x)
            new_value += z * (diffusion_betas[t] * (1 - alpha_head_t_minus_one) / (1 - alpha_head))
            alpha_head_t_minus_one = alpha_head

        # if T-1 > t > 0:
        #     new_value = new_value + z * torch.sqrt(diffusion_betas[t+1])

        img = new_value

        new_value = torch.clip(new_value, -1, 1) / 2 + 0.5

        ToPILImage()(noise_to_reduce[0]).save(out_path.joinpath(f"predicted_noise{t}.png"))
        ToPILImage()(new_value[0]).save(out_path.joinpath(f"predicted_new_value{t}.png"))


if __name__ == '__main__':
    test_prediction(
        model_path="../evaluation/diffusion_model/best_model.pt",
        image_path="~/Documents/data/aml/original128png/28594.png",
        out_path="~/Documents/data/aml/out"
    )
