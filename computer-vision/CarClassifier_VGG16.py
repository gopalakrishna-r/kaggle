import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


# reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


set_seed(31415)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    'input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True)

ds_valid_ = image_dataset_from_directory(
    'input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False)

#Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, tf.dtypes.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (
    ds_train_.
    map(convert_to_float).
    cache().
    prefetch(buffer_size=AUTOTUNE)
)

ds_valid = (
    ds_valid_.
    map(convert_to_float).
    cache().
    prefetch(buffer_size=AUTOTUNE)
)

# using a pre-trained model on ImageNet from keras applications module called VGG16
import matplotlib.pyplot as plt

pretrained_base = tf.keras.models.load_model('input/vgg16-pretrained-base')
pretrained_base.trainable = False

# attach head using layer of hidden units followed by a layer to transform the outputs
# to probability score for class 1 Truck.
# Flatten layer transforms the two dimensional outputs of the base to 1D inputs needed by the head
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history = model.fit(
    ds_train,
    validation_data = ds_valid,
    epochs=30
)

import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.loc[:,['val_loss', 'loss']].plot()
history_frame.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()