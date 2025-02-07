import os
import sys
import pickle
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import cv2

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from config import Config
from src.data.preprocessing import calculate_iou_score, process_data_for_rcnn


def prepare_dataset(dataset, save_path, max_samples=80000):
    os.makedirs(save_path, exist_ok=True)

    if len(os.listdir(save_path)) >= max_samples:
        print("Data Already Prepared.")
        return

    count = 0
    print(f"Processing dataset to {save_path}...")

    for image, annot in tqdm(dataset):
        image = np.array(image)
        boxes_annots = annot["annotation"]["object"]
        if not isinstance(boxes_annots, list):
            boxes_annots = [boxes_annots]

        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()[:Config.MAX_SELECTIONS]
        rects = np.array([[x, y, x+w, y+h] for x, y, w, h in rects])

        images, classes = process_data_for_rcnn(
            image, rects, Config.LABEL_MAP, boxes_annots
        )

        for idx, (img, label) in enumerate(zip(images, classes)):
            save_file = os.path.join(save_path, f"img_{count}_{idx}.pkl")
            with open(save_file, "wb") as pkl:
                pickle.dump({"image": img, "label": label}, pkl)

        count += 1
        if count >= max_samples:
            break

def main():
    os.makedirs(os.path.join(Config.DATA_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(Config.DATA_DIR, "val"), exist_ok=True)

    print("Downloading VOC dataset...")
    voc_dataset_train = torchvision.datasets.VOCDetection(
        root=Config.DATA_DIR,
        image_set="train",
        download=True,
        year="2007"
    )

    voc_dataset_val = torchvision.datasets.VOCDetection(
        root=Config.DATA_DIR,
        image_set="val",
        download=True,
        year="2007"
    )

    print("Processing training dataset...")
    prepare_dataset(
        voc_dataset_train,
        os.path.join(Config.DATA_DIR, "train"),
        max_samples=50000
    )

    print("Processing validation dataset...")
    prepare_dataset(
        voc_dataset_val,
        os.path.join(Config.DATA_DIR, "val"),
        max_samples=10000
    )

if __name__ == "__main__":
    main()
