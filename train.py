# train.py

import numpy as np
import tensorflow as tf
from config import *
from model import full_model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from data_loader import CustomImageLoader


# Data generators
train_gen = CustomImageLoader(train_imgs, train_lbls, IMAGES_FOLDER, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
val_gen = CustomImageLoader(val_imgs, val_lbls, IMAGES_FOLDER, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
test_gen = CustomImageLoader(test_imgs, test_lbls, images_folder, mode="test")
# Learning rate decay function
def decay_lr(epoch, curr_lr):
    if epoch % 10 == 0 and epoch != 0:
        return curr_lr * 0.1
    return curr_lr

# Callbacks
lr_decay_cb = LearningRateScheduler(decay_lr)
save_best_cb = ModelCheckpoint("logs/best_resnet_vit_model.keras", monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

# Training the model
full_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[lr_decay_cb, save_best_cb]
)
