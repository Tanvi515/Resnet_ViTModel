# config.py
BASE_DATA_PATH = "../input/stanford40"
IMAGES_FOLDER = os.path.join(BASE_DATA_PATH, "JPEGImages")
SPLITS_FOLDER = os.path.join(BASE_DATA_PATH, "ImageSplits")

BATCH_SIZE = 16
IMG_SIZE = (448, 448)
EPOCHS = 50
NUM_CLASSES = 40
MIXUP_ALPHA = 0.4
