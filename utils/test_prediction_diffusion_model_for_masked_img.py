from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from network.segmentation.unet_with_embedding import UNet
from utils.config_parser import ConfigParser


def do_multiple_diffusion_steps(img_0, t, diffusion_betas):
    alpha_t = torch.prod(1 - diffusion_betas[:t])
    return torch.normal(alpha_t * img_0, (1 - alpha_t))


@torch.no_grad()
def test_prediction(
        model_path="../train/best_model_12_epochs.pt",
        image_path="D:/aml/localData/masked128png/00000_Mask.png",
        label_path="D:/aml/localData/seg_mask128png/00000_Mask.png",
        out_path="~\\Documents\\data\\aml\\out",
        U=3
):
    config = ConfigParser.read("../configs/debugging_diffusion_model.yaml")

    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()
    out_path = Path(out_path).expanduser()

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

    img_orig = ToTensor()(Image.open(image_path)).to(device)[None, :]
    seg_mask = ToTensor()(Image.open(label_path)).to(device)[None, :]
    seg_mask = torch.cat([seg_mask] * 3, dim=1)

    img_shape = list(img_orig.shape)
    img = torch.randn(img_shape).to(device)

    alpha_head_t_minus_one = 0

    for t in tqdm(range(T-1, -1, -1)):
        for u in range(U):
            alpha_head = torch.prod(1 - diffusion_betas[:t]).to(device)
            alpha = torch.sqrt(1 - diffusion_betas[t]).to(device)

            noise_to_reduce = model.forward(img, torch.tensor([t]).to(device))
            new_value = 1 / torch.sqrt(alpha) * (img - (1 - alpha) / torch.sqrt(1 - alpha_head) * noise_to_reduce)

            if t > 0:
                z = torch.randn_like(img_orig)
                new_value += z * (diffusion_betas[t] * (1 - alpha_head_t_minus_one) / (1 - alpha_head))
                alpha_head_t_minus_one = alpha_head

            if t > 0:
                new_value[seg_mask == 0] = do_multiple_diffusion_steps(img_orig, t, diffusion_betas)[seg_mask == 0].to(device)

                if u < U:
                    new_value = torch.normal(torch.sqrt(1 - diffusion_betas[t]) * new_value, diffusion_betas[t]).to(device)
            else:
                new_value[seg_mask == 0] = img_orig[seg_mask == 0]

            img = new_value

            ToPILImage()(noise_to_reduce[0]).save(out_path.joinpath(f"predicted_noise{t}.png"))
            ToPILImage()(new_value[0]).save(out_path.joinpath(f"predicted_new_value{t}.png"))


if __name__ == '__main__':
    test_prediction(
        model_path="../evaluation/dm/best_model.pt",
        image_path="~\\Documents\\data\\aml\\original128png\\28594.png",
        label_path="~\\Documents\\data\\aml\\seg_mask128png\\28594.png",
        out_path="~\\Documents\\data\\aml\\out"
    )
