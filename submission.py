import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

BATCH_SIZE = 32 # Example batch size
RANDOM_SEED = 42


# Use the train and test splits provided by fashion-mnist. 
# x = images, y = labels
(x_val, y_val), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()



# Use the last 12000 samples of the training data as a validation set. 
validation_x = x_val[12000:]
validation_y = y_val[12000:]

# Use the first 48000 samples of the training data as the new training set.
train_x = x_val[:12000]
train_y = y_val[:12000]

# TODO normalization



'''Layer Specifications:
    2D convolutional layer, 28 filters, 3x3 window size, ReLU activation
    2x2 max pooling
    2D convolutional layer, 56 filters, 3x3 window size, ReLU activation
    fully-connected layer, 56 nodes, ReLU activation
    fully-connected layer, 10 nodes, softmax activation'''

model = Sequential([
    Conv2D(filters=28, kernel_size=3, activation='relu', input_shape=(28, 28, 1)), # CONVO LAYERS
    MaxPooling2D((2, 2)),
    Conv2D(filters=56, kernel_size=3, activation='relu'), # CONVO LAYERS, PATTERN DETECTION
    Flatten(), # why do we need this?
    Dense(units=56, activation='relu'),
    Dense(units=10, activation='softmax')
])

model.compile(
    optimizer=Adam(), 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)
# Train for 10 epochs
data = model.fit(train_x, train_y, epochs=10, batch_size=32,validation_data=(validation_x, validation_y))
model.summary() 

# TODO plotting training and validation accuracy/loss curves 
