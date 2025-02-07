import cv2
import numpy as np
from src.config import Config

def calculate_iou_score(box_1, box_2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box_1
    box2_x1, box2_y1, box2_x2, box2_y2 = box_2

    x1 = np.maximum(box1_x1, box2_x1)
    y1 = np.maximum(box1_y1, box2_y1)
    x2 = np.minimum(box1_x2, box2_x2)
    y2 = np.minimum(box1_y2, box2_y2)

    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection)

def process_data_for_rcnn(image, rects, class_map, boxes_annots):
    true_classes = []
    image_sections = []
    true_count = 0
    false_count = 0

    for annot in boxes_annots:
        label = annot["name"]
        box = [int(c) for _, c in annot["bndbox"].items()]
        box = np.array(box)

        for rect in rects:
            iou_score = calculate_iou_score(rect, box)
            if iou_score > Config.MAX_IOU_THRESHOLD:
                if true_count < Config.MAX_BOXES//2:
                    true_classes.append(class_map[label])
                    x1, y1, x2, y2 = rect
                    img_section = image[y1:y2, x1:x2]
                    image_sections.append(img_section)
                    true_count += 1
            else:
                if false_count < Config.MAX_BOXES//2:
                    true_classes.append(0)
                    x1, y1, x2, y2 = rect
                    img_section = image[y1:y2, x1:x2]
                    image_sections.append(img_section)
                    false_count += 1

    return image_sections, true_classes
