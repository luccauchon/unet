import os

os.environ['basedir_a'] = 'C:/temp2/'

from model import *
from luccauchon.data.Generators import AmateurDataFrameDataGenerator
import luccauchon.data.Generators as generators

df_train, df_val = generators.amateur_train_val_split('G:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/', number_elements=16)

dim_image = (256, 256, 3)
model = unet(input_size=dim_image)

train_generator = AmateurDataFrameDataGenerator(df_train, batch_size=4, dim_image=dim_image)
val_generator = AmateurDataFrameDataGenerator(df_val, batch_size=4, dim_image=dim_image)

model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=10, verbose=1, callbacks=None,
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
