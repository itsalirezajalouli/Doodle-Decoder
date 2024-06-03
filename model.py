#   Imports

import sys
import numpy as np
import tensorflow as tf

#   Load MNIST handwriting dataset

#mnist = tf.keras.datasets.mnist

path = 'mnist.npz'

with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']

#   Normalizing Manually (0,255) -> (0,1)

x_train, x_test = x_train / 255.0, x_test / 255.0

#   One-Hot Encoding

y_train = tf.keras.utils.to_categorical(y_train)

y_test = tf.keras.utils.to_categorical(y_test)

#   Adding 1 dimention, number of images, width, height, color channels

x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)

x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

#   Create a convolutional neural network

model = tf.keras.models.Sequential([

    #   Convolutional Layer, learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)
    ),

    #   Max-pooling layet, using 2x2 pool size (to reduce dimentionality)
    tf.keras.layers.MaxPooling2D(pool_size = (2, 2)),

    #   Flatten neurons (units) cause it should be a vector before feeding to fully connected layers
    tf.keras.layers.Flatten(),

    #   Add a hidden layer with dropout
    tf.keras.layers.Dense(128, activation = 'relu'),
    #   Prevent over-fitting
    tf.keras.layers.Dropout(0.5),

    #   Output Layer
    tf.keras.layers.Dense(10, activation = 'softmax')

])

#   Train neural network

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(x_train, y_train, epochs = 10)

#   Evaluate model's performance

model.evaluate(x_test, y_test, verbose = 2)

#   Save model to file

if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f'Model saved to {filename}.')