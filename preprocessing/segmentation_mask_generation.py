from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage.morphology import binary_opening, binary_dilation


def open_image_as_rgb(img_file):
    img = np.array(Image.open(img_file))
    return img / 255


def generate_segmentation_mask(img, img_facemask, th=0.05):
    """
    Generate a segmentation mask for the facemask using an original image (size 128x128)
    and the corresponding image with a synthetic facemask (size 128x128)
    """
    diff = np.abs(img - img_facemask)

    # plt.imshow(diff)
    # plt.show()

    img_bin = (diff > th).astype(np.uint8)

    # plt.imshow(img_bin)
    # plt.show()

    img_bin = binary_fill_holes(img_bin)

    img_opened = binary_opening(img_bin)

    # plt.imshow(img_opened)
    # plt.show()

    img_no_holes = binary_fill_holes(img_opened)

    # plt.imshow(img_no_holes)
    # plt.show()

    img_labeled = label(img_no_holes)
    out = (img_labeled == np.bincount(img_labeled.flatten())[1:].argmax() + 1).astype(int)

    # plt.imshow(out)
    # plt.show()

    # over segment the face mask
    out = binary_dilation(out)

    # plt.imshow(out)
    # plt.show()

    return out.astype(np.uint8)


def generate_all_segmentation_masks(orig_img_path, facemask_img_path, out_path):
    files = Path(facemask_img_path).expanduser().glob("*.png")
    orig_img_path = Path(orig_img_path).expanduser()
    out_path = Path(out_path).expanduser()

    for file in files:
        img_mask = open_image_as_rgb(file)
        img_orig = open_image_as_rgb(orig_img_path.joinpath(file.name.replace("_Mask.png", ".png")))

        out = generate_segmentation_mask(img_orig[:, :, 2], img_mask[:, :, 2])
        Image.fromarray(out).save(out_path.joinpath(file.name.replace("_Mask.png", "_seg.png")))

        # overlay = np.zeros_like(img_orig)
        # overlay[:, :, 1] = out
        # plt.imshow(cv2.addWeighted(img_mask, 0.5, overlay, 0.3, 0))
        # plt.show()


if __name__ == '__main__':
    generate_all_segmentation_masks(
        "~\\Documents\\data\\aml\\img_orig",
        "~\\Documents\\data\\aml\\img_facemask_downsampled",
        "~\\Documents\\data\\aml\\img_seg_mask"
    )
