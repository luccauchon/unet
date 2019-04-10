import luccauchon.data.__MYENV__ as E
import logging

E.APPLICATION_LOG_LEVEL = logging.INFO

import os

os.environ['basedir_a'] = 'F:/Temp2/'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import keras
import PIL
import numpy as np
import scipy

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

print('keras.__version__=' + str(keras.__version__))
print('tf.__version__=' + str(tf.__version__))
print('PIL.__version__=' + str(PIL.__version__))
print('np.__version__=' + str(np.__version__))
print('scipy.__version__=' + str(scipy.__version__))
print('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')
import sys
from pathlib import Path

print('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')

from model import *
from luccauchon.data.Generators import AmateurDataFrameDataGenerator
import luccauchon.data.Generators as generators


dim_image = (64, 64, 3)
batch_size = 32
model = unet(input_size=dim_image)
classes_id = [1]
number_elements = 256

if number_elements is not None:
    assert number_elements > batch_size
df_train, df_val = generators.amateur_train_val_split('F:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/', class_ids=classes_id, number_elements=number_elements)


train_generator = AmateurDataFrameDataGenerator(df_train, classes_id=classes_id, batch_size=batch_size, dim_image=dim_image)
val_generator = AmateurDataFrameDataGenerator(df_val, classes_id=classes_id, batch_size=batch_size, dim_image=dim_image)

modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath='F:/Temp2/unet_amateur_weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                  monitor='val_loss',
                                                  verbose=0, save_best_only=False, save_weights_only=False,
                                                  mode='auto', period=1)
reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1,
                                                      mode='auto', min_delta=0.001, cooldown=0, min_lr=10e-7)

assert len(df_train)//batch_size == train_generator.__len__()
assert len(df_val)//batch_size == val_generator.__len__()

model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.__len__(), epochs=10, verbose=1,
                    callbacks=[reduceLROnPlateau],#, modelCheckpoint],
                    validation_data=val_generator, validation_steps=val_generator.__len__(), class_weight=None, max_queue_size=10,
                    workers=8, use_multiprocessing=False, shuffle=True, initial_epoch=0)
