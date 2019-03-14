import os

os.environ['basedir_a'] = 'C:/temp2/'

from model import *
from luccauchon.data.Generators import AmateurDataFrameDataGenerator
from luccauchon.data.Generators import TrainDecoratorAmateurDataFrameDataGenerator
from luccauchon.data.Generators import ValDecoratorAmateurDataFrameDataGenerator

import luccauchon.data.C as C

df = C.generate_X_y_information_from_via_dataset('G:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/', number_elements=64)
dim_image = (256, 256, 3)
model = unet(input_size=dim_image)

generator = AmateurDataFrameDataGenerator(df, epochs=10, batch_size=4, img_w=dim_image[0], img_h=dim_image[1])
model.fit_generator(generator=TrainDecoratorAmateurDataFrameDataGenerator(generator), steps_per_epoch=None, epochs=10, verbose=1, callbacks=None,
                    validation_data=ValDecoratorAmateurDataFrameDataGenerator(generator), validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)
