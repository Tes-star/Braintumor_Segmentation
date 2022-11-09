import keras as keras
import numpy as np
import nibabel as nib
import itk
import itkwidgets
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
from skimage.util import montage
from skimage.transform import rotate
import os
import seaborn as sns
import keras
import keras.backend as K
from keras.callbacks import CSVLogger
import tensorflow as tf
from keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
# from keras.layers.experimental import preprocessing
import cv2
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=55620973696)])
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

image_count = 20


def import_images(path):
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


# def convert_data_2D(data):
#
#     #data_new = np.empty((image_count,240, 240,155),dtype=np.int32)
#
#
#     x_train_ten = tf.convert_to_tensor(train_x)
#     x_train_ten = tf.convert_to_tensor(train_y)
from keras.utils import to_categorical

path = './brain_images/'
x, y = import_images(path)

x_pre = np.zeros((image_count, 240, 240, 160), dtype=np.int32)
x_pre[:, :, :, 1:156] = x
x = x_pre
del x_pre
x = tf.convert_to_tensor(x,dtype=tf.int32)

y_pre = np.zeros((image_count, 240, 240, 160), dtype=np.int32)
y_pre[:, :, :, 1:156] = y
y = y_pre
del y_pre
# train_y = to_categorical(train_y, num_classes=5)
y = to_categorical(y, num_classes=5)
y = tf.convert_to_tensor(y,dtype=tf.int32)

# simple 3d
# create 2D CNN model
def get_3D_model():
    """Build a 3D convolutional neural network model."""

    ker_init = 'he_normal'

    inputs = keras.Input(shape=(240, 240, 160, 1))

    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)

    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
   # x = BatchNormalization()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
   # x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=2)(x)

    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool3D(pool_size=1)(x)

    # x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
    x = UpSampling3D()(x)
    x = Conv3D(filters=16, kernel_size=(3), activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=16, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)

    x = UpSampling3D()(x)
    x = Conv3D(filters=8, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    x = BatchNormalization()(x)
    x = Conv3D(filters=4, kernel_size=3, activation="relu", kernel_initializer=ker_init, padding="same")(x)
    #x = BatchNormalization()(x)

    # x = Conv2DTranspose(filters=1, kernel_size=3, activation="softmax",padding="same")(x)
    # x = Conv2DTranspose(filters=1, kernel_size=3)(x)
    x = Conv3D(5, (1, 1, 1), activation='softmax')(x)
    outputs = x

    # outputs = layers.Dense(units=240*240*1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_3D_model()
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

model.fit(x[:15], y[:15],batch_size=1,  epochs=100,
          validation_data=(x[15:], y[15:])
         )
print(x.shape)
print(np.unique(x))
