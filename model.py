# model.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Reshape, GlobalAveragePooling1D, AveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import average_precision_score

l
class TinyViT(tf.keras.Model):
    def __init__(self, patches, dim, heads, layers, ff_dim, dropout=0.1):
        super(TinyViT, self).__init__()
        self.patches = patches
        self.dim = dim
        self.heads = heads
        self.layers = layers
        self.ff_dim = ff_dim
        self.dropout = dropout
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.dim)
        self.dense1 = tf.keras.layers.Dense(self.ff_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.dim)

    def call(self, inputs):
        x = self.mha(inputs, inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = self.dense1(x)
        x = tf.keras.layers.LayerNormalization()(x)
        return self.dense2(x)

inp = tf.keras.Input(shape=(448, 448, 3))

resnet = ResNet50(include_top=False, input_tensor=inp)
feat_output = resnet.get_layer("conv5_block3_out").output

vit_14 = TinyViT(196, 2048, heads=4, layers=2, ff_dim=4096)
vit_out_14 = vit_14(Reshape((196, 2048))(feat_output))
reconv_14 = Reshape((14, 14, 2048))(vit_out_14)
pooled_7 = AveragePooling2D(pool_size=2, strides=2, padding='same')(reconv_14)


vit_7 = TinyViT(49, 2048, heads=4, layers=2, ff_dim=4096)
vit_out_7 = vit_7(Reshape((49, 2048))(pooled_7))

gap = GlobalAveragePooling1D()(vit_out_7)
dense_1 = Dense(1024, activation='relu')(gap)
drop = Dropout(0.5)(dense_1)
final_out = Dense(NUM_CLASSES, activation='softmax')(drop)

full_model = Model(inputs=inp, outputs=final_out)
l
opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=3e-5)
full_model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

full_model.summary()
