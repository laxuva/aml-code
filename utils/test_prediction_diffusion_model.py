from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from network.segmentation.unet import UNet
from utils.config_parser import ConfigParser


def do_multiple_diffusion_steps(img_0, t, diffusion_betas):
    alpha_t = torch.prod(1 - diffusion_betas[:t])
    return torch.normal(alpha_t * img_0, (1 - alpha_t))


def test_prediction(
        model_path="../train/best_model_12_epochs.pt",
        image_path="D:/aml/localData/masked128png/00000_Mask.png",
        label_path="D:/aml/localData/seg_mask128png/00000_Mask.png",
        out_path="~\\Documents\\data\\aml\\out"
):
    config = ConfigParser.read("../configs/debugging_diffusion_model.yaml")
    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()
    out_path = Path(out_path).expanduser()

    diffusion_betas = torch.linspace(
        config["training"]["diffusion_beta_1"],
        config["training"]["diffusion_beta_capital_t"],
        config["training"]["diffusion_steps"]
    )

    T = config["training"]["diffusion_steps"]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = UNet(**config["model"]["params"])
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    x = ToTensor()(Image.open(image_path)).to(device)[None, :]
    seg_mask = ToTensor()(Image.open(label_path))[None, :]
    seg_mask = torch.cat((seg_mask, seg_mask, seg_mask), dim=1)

    img_t = do_multiple_diffusion_steps(x, T, diffusion_betas)
    img_t[seg_mask != 0] = torch.normal(0.5, 0.02, img_t[seg_mask != 0].size())

    for t in range(T, 0, -1):
        ToPILImage()(img_t[0]).save(out_path.joinpath(f"orig_img_{t}.png"))

        img_t_minus_1_pred = model.forward(img_t)

        # img_t in next step
        img_t = do_multiple_diffusion_steps(x, t - 1, diffusion_betas)
        img_t[seg_mask != 0] = img_t_minus_1_pred[seg_mask != 0].detach()

        ToPILImage()(img_t[0]).save(out_path.joinpath(f"img_{t}.png"))

    img_0_pred = model.forward(img_t)

    ToPILImage()(img_0_pred[0]).save(out_path.joinpath("img_0.png"))

    img_final = x.clone()
    img_final[seg_mask != 0] = img_0_pred[seg_mask != 0]
    ToPILImage()(img_final[0]).save(out_path.joinpath("img_0_final.png"))


if __name__ == '__main__':
    test_prediction(
        model_path="../train/final_model.pt",
        image_path="~\\Documents\\data\\aml\\masked128png\\28594.png",
        label_path="~\\Documents\\data\\aml\\seg_mask128png\\28594.png",
        out_path="~\\Documents\\data\\aml\\out"
    )
