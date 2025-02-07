import os

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    MODEL_DIR = os.path.join(ROOT_DIR, "models")

    # Dataset parameters
    MAX_IOU_THRESHOLD = 0.7
    MAX_BOXES = 50
    MAX_SELECTIONS = 1000

    # Model parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 5

    # Inference parameters
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    MAX_INFERENCE_SELECTIONS = 300

    # Image parameters
    IMAGE_SIZE = (224, 224)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

    # Label mappings
    LABEL_MAP = {
        'bg': 0,               # Background class
        'train': 1,
        'boat': 2,
        'bus': 3,
        'aeroplane': 4,
        'car': 5,
        'sofa': 6,
        'chair': 7,
        'pottedplant': 8,
        'diningtable': 9,
        'person': 10,
        'sheep': 11,
        'horse': 12,
        'cow': 13,
        'bottle': 14,
        'tvmonitor': 15,
        'dog': 16,
        'bicycle': 17,
        'motorbike': 18,
        'cat': 19,
        'bird': 20
    }

    IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
