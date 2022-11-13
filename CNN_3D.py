import os

import keras
import nibabel as nib
import numpy as np
from keras.layers import Conv3D, BatchNormalization, MaxPool3D, UpSampling3D
from keras.optimizers import Adam
from keras.utils import to_categorical


def import_images(path,image_count):

    x = np.empty((image_count, 240, 240, 155), dtype=np.int32)
    y = np.empty((image_count, 240, 240, 155), dtype=np.int32)

    # Liste aller Dateien in annotation_folder erstellen
    folders = os.listdir(path)

    # Aus Liste files .hdr Dateien löschen
    i = 0

    for image_folder in folders[:image_count]:
        image_path = path + '/' + image_folder + '/' + image_folder
        x[i] = nib.load(image_path + '_flair.nii.gz').get_fdata()
        y[i] = nib.load(image_path + '_seg.nii.gz').get_fdata()
        i = i + 1
        # print(i)
    return x, y


def get_3D_model_old():
    """Build a 3D convolutional neural network model."""

    ker_init = 'he_normal'

    inputs = keras.Input(shape=(240, 240, 160, 1))

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
    x = Conv3D(filters=8, kernel_size=(3), activation="relu", kernel_initializer=ker_init, padding="same")(x)
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
    x = Conv3D(5, (1, 1, 1), activation='softmax')(x)
    outputs = x

    # outputs = layers.Dense(units=240*240*1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


if __name__ == "__main__":
    image_count =5
    path = './brain_images/'
    x, y = import_images(path,image_count)

    x_pre = np.zeros((image_count, 240, 240, 160), dtype=np.int32)
    x_pre[:, :, :, 1:156] = x
    x = x_pre

    y_pre = np.zeros((image_count, 240, 240, 160), dtype=np.int32)
    y_pre[:, :, :, 1:156] = y
    y = y_pre

    # train_y = to_categorical(train_y, num_classes=5)
    y = to_categorical(y, num_classes=5)

    model = get_3D_model_old()
    model.summary()

    from keras.metrics import MeanIoU, OneHotMeanIoU
    # from sklearn.metrics import recall_score, f1_score, precision_score
    from keras import metrics

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001),
                  metrics=['accuracy', 'categorical_accuracy', OneHotMeanIoU(num_classes=5)
                           ]
                  # metrics = ['accuracy','sparse_categorical_accuracy','categorical_accuracy','categorical_crossentropy',#MeanIoU(num_classes=5)
                  #                     ]
                  )

    # tf.keras.metrics.MeanIoU(num_classes=2)]

    model.fit(x[:image_count-2], y[:image_count-2], batch_size=1, epochs=100,
              validation_data=(x[image_count-2:], y[image_count-2:]))
    print(x.shape)
    print(np.unique(x))
