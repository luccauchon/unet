import luccauchon.data.__MYENV__ as E
import logging

E.APPLICATION_LOG_LEVEL = logging.INFO
from loguru import logger
from multiprocessing import freeze_support
import os
if os.name is not 'nt':
    os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
else:
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

logger.debug('keras.__version__=' + str(keras.__version__))
logger.debug('tf.__version__=' + str(tf.__version__))
logger.debug('PIL.__version__=' + str(PIL.__version__))
logger.debug('np.__version__=' + str(np.__version__))
logger.debug('scipy.__version__=' + str(scipy.__version__))
logger.debug('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')
import sys
from pathlib import Path

logger.debug('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')

from model import *
from luccauchon.data.Generators import COCODataFrameDataGenerator


def start():
    categories = ['person']#, 'tie']
    if os.name is not 'nt':
        data_dir_source_coco = '/gpfs/home/cj3272/56/APPRANTI/cj3272/dataset/coco/'
    else:
        data_dir_source_coco = 'F:/APPRANTI/dataset/Dataset_COCO_Appranti_80_20__[1]/'
    img_h = 480
    img_w = 640
    c = 3
    batch_size = 4

    train_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, img_h=img_h, img_w=img_w, c=c, data_type_source_coco='train2019')
    val_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, img_h=img_h, img_w=img_w, c=c, data_type_source_coco='val2019')

    model = unet(input_size=(img_h, img_w, c))

    modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath='F:/APPRANTI/Modeles/UNET/unet_coco_weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                      monitor='val_loss',
                                                      verbose=0, save_best_only=False, save_weights_only=False,
                                                      mode='auto', period=1)
    reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1,
                                                          mode='auto', min_delta=0.001, cooldown=0, min_lr=10e-7)

    model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=30, verbose=1,
                        callbacks=[reduceLROnPlateau, modelCheckpoint],
                        validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                        workers=8, use_multiprocessing=(True if os.name is not 'nt' else False), shuffle=True, initial_epoch=0)


if __name__ == '__main__':
    freeze_support()
    start()