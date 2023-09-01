from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import os

from tqdm import tqdm

# np.set_printoptions(threshold=sys.maxsize)

SEG_DATASET_DIR_PATH = "/Data1/jbchae/U2PL/data/VOC2012/SegmentationClass"


def main():
    num_pixel = np.zeros(shape=22)
    load_image_fname_tqdm = tqdm(
        enumerate(os.listdir(SEG_DATASET_DIR_PATH)),
        desc=f"Load image... _ / {len(os.listdir(SEG_DATASET_DIR_PATH))}",
        total=len(os.listdir(SEG_DATASET_DIR_PATH)),
    )
    for i, img_fname in load_image_fname_tqdm:
        img = Image.open(os.path.join(SEG_DATASET_DIR_PATH, img_fname))
        pixels = list(img.getdata())
        w, h = img.size
        pixels = [pixels[i * w : (i + 1) * w] for i in range(h)]

        for row in pixels:
            for pixel in row:
                if pixel == 255:
                    num_pixel[-1] += 1
                    continue
                num_pixel[pixel] += 1
        load_image_fname_tqdm.set_description(
            desc=f"Load image... {i + 1} / {len(os.listdir(SEG_DATASET_DIR_PATH))}"
        )
    num_pixel = num_pixel[:-1]
    num_pixel_without_bg = num_pixel[1:]

    num_pixel_norm = [i / np.sum(num_pixel) for i in num_pixel]
    num_pixel_without_bg_norm = [
        i / np.sum(num_pixel_without_bg) for i in num_pixel_without_bg
    ]
    print(num_pixel_norm)
    print(num_pixel_without_bg_norm)

    plt.xticks(ticks=list(range(len(num_pixel))), labels=list(range(len(num_pixel))))
    plt.plot(list(range(len(num_pixel))), num_pixel_norm)

    plt.tight_layout()

    plt.savefig(f"fig/pixel_distribution.png")
    np.save(file=f"npy/pixel_distribution.npy", arr=num_pixel)
    plt.clf()
    plt.cla()
    plt.close()
    
    plt.xticks(ticks=list(range(len(num_pixel_without_bg))), labels=list(range(len(num_pixel_without_bg))))
    plt.plot(list(range(len(num_pixel_without_bg))), num_pixel_without_bg_norm)

    plt.tight_layout()

    plt.savefig(f"fig/pixel_distribution_without_bg.png")
    np.save(file=f"npy/pixel_distribution_without_bg.npy", arr=num_pixel_without_bg)
    plt.clf()
    plt.cla()
    plt.close()


if __name__ == "__main__":
    main()
