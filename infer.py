import logging
import os
import sys
import time
from argparse import ArgumentParser

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision.transforms import ToPILImage
import yaml
from PIL import Image
from tqdm import tqdm

from u2pl.models.model_helper import ModelBuilder
from u2pl.utils.utils import (
    AverageMeter,
    check_makedirs,
    colorize,
    convert_state_dict,
    intersectionAndUnion,
)


# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/psp_best.pth",
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder", type=str, default="viewer", help="results save folder"
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]
    crop_size = cfg_dset["val"]["crop"]["size"]
    crop_h, crop_w = crop_size

    assert num_classes > 1

    os.makedirs(args.save_folder, exist_ok=True)
    gray_folder = (
        os.path.join(*[args.save_folder, "color", *args.config.split(sep="/")[1:4]])
        if len(args.config.split(sep="/")) > 4
        else os.path.join(args.save_folder, "color")
    )
    os.makedirs(gray_folder, exist_ok=True)
    color_folder = (
        os.path.join(*[args.save_folder, "color", *args.config.split(sep="/")[1:4]])
        if len(args.config.split(sep="/")) > 4
        else os.path.join(args.save_folder, "color")
    )
    os.makedirs(color_folder, exist_ok=True)

    # data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]
    data_list = []

    if "cityscapes" in cfg_dset["type"]:
        data_root, f_data_list = "data/cityscapes", "data/splits/cityscapes/val.txt"
        for line in open(f_data_list, "r"):
            arr = [
                line.strip(),
                "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    else:
        data_root, f_data_list = "data/VOC2012", "data/splits/pascal/val.txt"
        for line in open(f_data_list, "r"):
            arr = [
                "JPEGImages/{}.jpg".format(line.strip()),
                "SegmentationClassAug/{}.png".format(line.strip()),
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False
    logger.info("=> creating model from '{}' ...".format(args.model_path))

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])
    checkpoint = torch.load(args.model_path)
    key = "teacher_state" if "teacher_state" in checkpoint.keys() else "model_state"
    logger.info(f"=> load checkpoint[{key}]")

    saved_state_dict = convert_state_dict(checkpoint[key])
    model.load_state_dict(saved_state_dict, strict=False)
    model.cuda()
    logger.info("Load Model Done!")

    input_scale = [769, 769] if "cityscapes" in data_root else [513, 513]
    colormap = create_pascal_label_colormap()
    label_names = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "dining table",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    model.eval()
    for image_path, label_path in tqdm(data_list):
        image_name = image_path.split("/")[-1]

        org_image = Image.open(image_path).convert("RGB")
        org_image = np.asarray(org_image).astype(np.float32)
        h, w, _ = org_image.shape
        image = (org_image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.unsqueeze(dim=0)
        image = F.interpolate(image, input_scale, mode="bilinear", align_corners=True)

        gt = Image.open(
            os.path.join(
                "/Data1/jbchae/U2PL/data/VOC2012/SegmentationClass",
                os.path.splitext(image_name)[0] + ".png",
            )
        ).convert("RGB")
        gt = np.asarray(gt).astype("uint8")

        output = net_process(model, image)
        output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # color_mask = Image.fromarray(colorful(mask, colormap))
        # color_mask.save(os.path.join(color_folder, image_name))
        color_mask, labels = colorful(mask, colormap)

        fig = plt.figure()

        gs = GridSpec(2, 2)

        ax1 = fig.add_subplot(gs[0, :])
        ax1.imshow(org_image.astype("uint8"))
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.imshow(gt)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[1, 1])
        ax3.imshow(color_mask.astype("uint8"))
        ax3.axis("off")

        norm_colormap = colormap / 255.0
        fig.legend(
            handles=[
                mpatches.Patch(color=norm_colormap[label], label=f"{label_names[label]}")  # type: ignore
                for label in labels
            ]
        )

        fig.tight_layout()

        fig.savefig(os.path.join(color_folder, image_name))
        plt.cla()
        plt.clf()
        plt.close()


def colorful(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    labels = list()
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
        if i not in labels:
            labels.append(i)

    return np.uint8(color_mask), labels


def create_pascal_label_colormap():
    """Creates a label colormap used in Pascal segmentation benchmark.
    Returns:
        A colormap for visualizing segmentation results.
    """
    colormap = 255 * np.ones((21, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]
    colormap[20] = [0, 64, 128]

    return colormap


@torch.no_grad()
def net_process(model, image):
    input = image.cuda()
    output = model(input)["pred"]
    return output


if __name__ == "__main__":
    main()
