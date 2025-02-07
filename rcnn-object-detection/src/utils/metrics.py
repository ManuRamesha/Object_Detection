import numpy as np

def non_max_suppression(boxes, scores, labels, threshold=0.5, iou_threshold=0.5):
    idxs = np.where(scores > threshold)[0]
    boxes = boxes[idxs]
    scores = scores[idxs]
    labels = labels[idxs]

    idxs = np.argsort(scores)

    chosen_boxes = []
    chosen_scores = []
    chosen_labels = []

    while len(idxs) > 0:
        last = len(idxs) - 1
        chosen_idx = idxs[last]
        chosen_box = boxes[chosen_idx]
        chosen_score = scores[chosen_idx]
        chosen_label = labels[chosen_idx]

        chosen_boxes.append(chosen_box)
        chosen_scores.append(chosen_score)
        chosen_labels.append(chosen_label)

        idxs = np.delete(idxs, last)

        i = len(idxs) - 1
        while i >= 0:
            idx = idxs[i]
            curr_box = boxes[idx]
            curr_label = labels[idx]

            if (curr_label == chosen_label and
                calculate_iou_score(curr_box, chosen_box) > iou_threshold):
                idxs = np.delete(idxs, i)
            i -= 1

    return chosen_boxes, chosen_scores, chosen_labels
