import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_boxes(img, boxes, scores, labels, class_map=None):
    nums = len(boxes)
    img_copy = img.copy()
    for i in range(nums):
        x1y1 = tuple(np.array(boxes[i][0:2]).astype(np.int32))
        x2y2 = tuple(np.array(boxes[i][2:4]).astype(np.int32))
        img_copy = cv2.rectangle(img_copy, x1y1, x2y2, (255, 0, 0), 2)
        label = int(labels[i])
        label_txt = class_map[label] if class_map is not None else str(label)
        img_copy = cv2.putText(
            img_copy,
            "{} {:.4f}".format(label_txt, scores[i]),
            x1y1,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 0, 255),
            2
        )
    return img_copy
