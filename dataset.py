import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
import re

base_data_path = "../input/stanford40"
images_folder = os.path.join(base_data_path, "JPEGImages")
splits_folder = os.path.join(base_data_path, "ImageSplits")

all_image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]

def get_label_from_filename(filename):
    matched = re.match(r"([a-zA-Z_]+)_\d+", filename)
    return matched.group(1) if matched else None

labels = [get_label_from_filename(f) for f in all_image_files]
train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
    all_image_files, labels, test_size=0.42, random_state=42, stratify=labels
)
train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
    train_imgs, train_lbls, test_size=0.10, random_state=42, stratify=train_lbls
)

label_map = {name: idx for idx, name in enumerate(all_classes)}

class CustomImageLoader(Sequence):
    def __init__(self, filenames, labels, root_dir, batch_size=16, img_size=(448, 448), mode="train"):
        self.filenames = filenames
        self.labels = labels
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode  

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_filenames = self.filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]
        X, y = [], []

        for fname, label in zip(batch_filenames, batch_labels):
            img_path = os.path.join(self.root_dir, fname)
            try:
                img_arr = img_to_array(load_img(img_path, target_size=self.img_size)) / 255.0
                X.append(img_arr)
                y.append(to_categorical(label_map[label], num_classes=40))
            except Exception as err:
                print(f"Failed to load {img_path}: {err}")
                continue

        return np.array(X), np.array(y)
train_gen = CustomImageLoader(train_imgs, train_lbls, images_folder, mode="train")
val_gen = CustomImageLoader(val_imgs, val_lbls, images_folder, mode="val")
test_gen = CustomImageLoader(test_imgs, test_lbls, images_folder, mode="test")
