import os
import shutil

import cv2

from tqdm import tqdm

root_dir = "C:/Users/admin/Downloads/PASCAL VOC 2012 outputs/color/pascal"

splits = ["92", "183", "366", "732", "1464"]

for split in splits:
    semi_dir_path = os.path.join(*[root_dir, split, "ours"])
    sup_dir_path = os.path.join(*[root_dir, split, "suponly"])

    con_dir_path = os.path.join(*[root_dir, split, "concat"])
    if os.path.exists(con_dir_path) is False:
        os.makedirs(con_dir_path)
    else:
        shutil.rmtree(con_dir_path)
        os.makedirs(con_dir_path)

    assert os.listdir(semi_dir_path) == os.listdir(
        sup_dir_path
    ), "File names in each directory must be same"
    read_img_tqdm = tqdm(
        zip(os.listdir(semi_dir_path), os.listdir(sup_dir_path)),
        desc="Concatenate image _ with _",
        total=len(os.listdir(semi_dir_path)),
        leave=False,
    )
    for semi_img_name, sup_img_name in read_img_tqdm:
        read_img_tqdm.set_description(
            f"Concatenate image {os.path.join(*[split, 'ours', semi_img_name])} with {os.path.join(*[split, 'suponly', sup_img_name])}"
        )
        semi_img = cv2.imread(os.path.join(semi_dir_path, semi_img_name))
        sup_img = cv2.imread(os.path.join(sup_dir_path, sup_img_name))

        con_img = cv2.hconcat([semi_img, sup_img])
        cv2.imwrite(filename=os.path.join(con_dir_path, semi_img_name), img=con_img)
