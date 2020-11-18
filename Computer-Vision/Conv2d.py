import tensorflow.keras as keras
import tensorflow.keras.layers as layers

model = keras.Sequential([
    layers.Conv2D(filters = 64, kernel_size=3), # no activation
])