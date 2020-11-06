import pandas as pd
import os
from IPython.display import display

RED_WINE_PATH = os.path.join("input", "dl-course-data")

csv_path = os.path.join(RED_WINE_PATH, "red-wine.csv")
red_wine = pd.read_csv(csv_path)

# Create training and validation splits
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
display(df_train.head(4))

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=[11]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
    ]
)

model.compile(optimizer = 'adam',
loss = 'mae'
)

history = model.fit(
X_train, y_train,
validation_data = (X_valid, y_valid),
batch_size = 256,
epochs = 100,
verbose = 1
)

import pandas as pd
import matplotlib.pyplot as plt
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
plt.show()
