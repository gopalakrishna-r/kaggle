from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer
    layers.Dense(units=1)
])

model.compile(
    optimizer='adam',
    loss='mae'
)
