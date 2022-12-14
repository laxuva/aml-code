from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from network.unet_with_embedding import UNet
from utils.config_parser import ConfigParser


def do_multiple_diffusion_steps(img, alpha_head, device="cuda"):
    e = torch.normal(0, 1, img.shape).to(device)
    return torch.sqrt(alpha_head) * img + torch.sqrt(1 - alpha_head) * e


@torch.no_grad()
def test_prediction(
        model: UNet,
        img_orig: torch.Tensor,
        seg_mask: torch.Tensor,
        T: int,
        U: int,
        diffusion_betas: np.ndarray,
        device: torch.device,
        out_path: Path = None,
        show_tqdm: bool = True
):
    img_shape = list(img_orig.shape)
    img_new = torch.randn(img_shape).to(device)

    alpha_head_t_minus_one = 0

    for t in tqdm(range(T)[::-1]) if show_tqdm else range(T)[::-1]:
        alpha_head = torch.prod(1 - diffusion_betas[:t + 1]).to(device)
        alpha = 1 - diffusion_betas[t].to(device)

        for u in range(U):
            noise_to_reduce = model.forward(img_new, torch.tensor([t]).to(device))
            # noise_to_print = noise_to_reduce[0] - torch.min(noise_to_reduce[0]) # beginning from 0
            # ToPILImage()(noise_to_print / torch.max(noise_to_print[0])).save(out_path.joinpath(f"predicted_noise{t}.png"))

            img_new = 1 / torch.sqrt(alpha) * (
                        img_new - diffusion_betas[t] * noise_to_reduce / torch.sqrt(1 - alpha_head))

            if t > 0:
                z = torch.randn_like(img_orig)
                img_new += z * torch.sqrt((diffusion_betas[t] * (1 - alpha_head_t_minus_one) / (1 - alpha_head)))
                img_new[seg_mask == 0] = do_multiple_diffusion_steps(img_orig, alpha_head, diffusion_betas)[
                    seg_mask == 0].to(device)

                if u < U:
                    img_new = torch.normal(torch.sqrt(1 - diffusion_betas[t]) * img_new, diffusion_betas[t]).to(device)




        if out_path is not None:
            ToPILImage()(torch.clip(img_new[0],
                                    -1,
                                    1) / 2 + 0.5).save(out_path.joinpath(f"predicted_new_value_without_masked_original{t}.png"))
            img_for_print = img_new.clone()
            img_for_print[seg_mask == 0] = img_orig[seg_mask == 0]
            ToPILImage()(torch.clip(img_for_print[0], -1, 1) / 2 + 0.5).save(out_path.joinpath(f"predicted_new_value{t}.png"))

        alpha_head_t_minus_one = alpha_head

    img_new[seg_mask == 0] = img_orig[seg_mask == 0]
    return torch.clip(img_new[0], -1, 1) / 2 + 0.5


def test_prediction_from_files(model_path, image_path, label_path, out_path, config_file="../configs/diffusion_model.yaml"):
    config = ConfigParser.read(config_file)

    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()
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
    U = config["evaluation"]["harmonization_steps"]

    model = UNet(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_orig = ToTensor()(Image.open(image_path)).to(device)[None, :] * 2 - 1
    seg_mask = ToTensor()(Image.open(label_path)).to(device)[None, :]
    seg_mask = torch.cat([seg_mask] * 3, dim=1)

    test_prediction(model, img_orig, seg_mask, T, U, diffusion_betas, device, out_path)


if __name__ == '__main__':
    test_prediction_from_files(
        model_path="../evaluation/diffusion_model/best_e150_w_arg.pt", # 32_l1_ep51_64-1024 best_e150_w_arg
        image_path="~/Documents/data/aml/original128png/32254.png",  # 00186 00048 00018 45844 00375 65959
        label_path="~/Documents/data/aml/seg_mask128png/32254.png",
        out_path="~/Documents/data/aml/out"
    )
