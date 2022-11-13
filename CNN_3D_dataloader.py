from keras.layers import Conv3D, BatchNormalization
from sklearn.model_selection import train_test_split

from CNN_3D import get_3D_model_old
from dataloader import DataGenerator, pathListIntoIds, train_and_val_directories, keras
import os

import keras
import nibabel as nib
import numpy as np
from keras.layers import Conv3D, BatchNormalization, MaxPool3D, UpSampling3D
from keras.optimizers import Adam
from keras.utils import to_categorical


def get_3D_model():
    """Build a 3D convolutional neural network model."""

    ker_init = 'he_normal'

    inputs = keras.Input(shape=(128, 128, 128, 2))

    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)

    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)

    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=1)(x)

    # x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = UpSampling3D()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)

    x = UpSampling3D()(x)
    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)

    # x = Conv2DTranspose(filters=1, kernel_size=3, activation="softmax",padding="same")(x)
    # x = Conv2DTranspose(filters=1, kernel_size=3)(x)
    x = Conv3D(4, (1, 1, 1), activation='softmax')(x)
    outputs = x

    # outputs = layers.Dense(units=240*240*1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


if __name__ == '__main__':
    train_and_test_ids = pathListIntoIds(train_and_val_directories);

    train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
    train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)

    training_generator = DataGenerator(train_ids)
    valid_generator = DataGenerator(val_ids)
    test_generator = DataGenerator(test_ids)

    model = get_3D_model()
    model.summary()

    from keras.metrics import MeanIoU, OneHotMeanIoU
    # from sklearn.metrics import recall_score, f1_score, precision_score
    from keras import metrics

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy', 'categorical_accuracy', OneHotMeanIoU(num_classes=4)
                           ]
                  # metrics = ['accuracy','sparse_categorical_accuracy','categorical_accuracy','categorical_crossentropy',#MeanIoU(num_classes=5)
                  #                     ]
                  )

    # tf.keras.metrics.MeanIoU(num_classes=2)]
    # Train model on dataset
    # model.fit_generator(generator=training_generator,
    #                     validation_data=valid_generator,
    #                     use_multiprocessing=True,
    #                     workers=6)
    model.fit(training_generator, batch_size=1, epochs=100,
              validation_data=valid_generator,use_multiprocessing=True,workers=1)
