from pathlib import Path

import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from train.diffusion_model.alternative.diffusion_model import SimpleUnet, sample_timestep, show_tensor_image, forward_diffusion_sample, get_betas

@torch.no_grad()
def sample_plot_image(model, img, seg_mask, U=3, T=300, device="cpu", epoch=None, img_size=128):
    # Sample noise
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T / num_images)
    img_new = torch.randn((1, 3, img_size, img_size), device=device)
    seg_mask = torch.cat([seg_mask]*3, dim=1)

    for i in tqdm(range(T)[::-1]):
        for u in range(U):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img_new = sample_timestep(img_new, t, model) # model prediction
            show_tensor_image(img_new[0].detach().cpu(), False).save(out_path.joinpath(f"predicted_image{i}.png"))

            if i > 0:
                img_new[seg_mask == 0] = forward_diffusion_sample(img, t, device)[0][seg_mask == 0]
                show_tensor_image(img_new[0].detach().cpu(), False).save(out_path.joinpath(f"combined_image{i}.png"))
                if u < U:
                    beta = get_betas().to(device)[t]
                    img_new = torch.normal(torch.sqrt(1 - beta) * img_new, beta).to(device)
            else:
                img_new[seg_mask == 0] = img[seg_mask == 0]
                show_tensor_image(img_new[0].detach().cpu(), False).save(out_path.joinpath(f"combined_image{i}.png"))

        if i % stepsize == 0:
            plt.subplot(1, num_images, i // stepsize + 1)
            plt.title(f"Step: {i}")
            show_tensor_image(img_new.detach().cpu())

    if epoch is not None:
        plt.suptitle(f"Epoch: {epoch}")
    plt.show()


if __name__ == '__main__':
    model_path = "../evaluation/dm/model.pt"
    image_path = "~\\Documents\\data\\aml\\masked128png\\00018.png"
    label_path = "~\\Documents\\data\\aml\\seg_mask128png\\00018.png"
    out_path = "~\\Documents\\data\\aml\\out"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SimpleUnet()
    model.to(device)
    model.load_state_dict(torch.load(Path(model_path),
                                     map_location=device))
    # model.eval()

    image_path = Path(image_path).expanduser()
    label_path = Path(label_path).expanduser()
    out_path = Path(out_path).expanduser()

    img = ToTensor()(Image.open(image_path)).to(device)[None, :]
    img = img * 2 - 1
    seg_mask = ToTensor()(Image.open(label_path)).to(device)[None, :]
    # ToPILImage()(seg_mask[0] * 255).show()

    sample_plot_image(model,
                      img,
                      seg_mask,
                      device=device)
