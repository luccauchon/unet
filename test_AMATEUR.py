import os

os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'
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
from model import *
from luccauchon.data.Generators import AmateurDataFrameDataGenerator
import luccauchon.data.Generators as generators
import luccauchon.data.C as C

print('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')

val = 1
dim_image = (256, 256, 3)
batch_size = 24

if val == 1:
    df_test = C.generate_X_y_raw_from_amateur_dataset('/gpfs/groups/gc056/APPRANTI/cj3272/dataset/22FEV2019/GEN_segmentation/', dim_image=dim_image, number_elements=batch_size)
    trained = keras.models.load_model('/gpfs/home/cj3272/56/APPRANTI/cj3272/unet/unet_weights.01-0.2285.hdf5')
    for i in range(0, len(df_test)):
        img = df_test.iloc[i]['the_image']
        img = np.expand_dims(img, axis=0)
        filename = df_test.iloc[i]['filename']
        results = trained.predict(x=img, verbose=1)
        import scipy.misc

        my_mask = results[0]
        assert isinstance(my_mask, np.ndarray)
        import PIL.Image as Image

        scipy.misc.imsave(filename + '_mask.jpg', Image.fromarray(my_mask[:, :, 0]))
else:
    df_test = generators.amateur_test('/gpfs/groups/gc056/APPRANTI/cj3272/dataset/22FEV2019/GEN_segmentation/', number_elements=batch_size)
    test_generator = AmateurDataFrameDataGenerator(df_test, batch_size=batch_size, dim_image=dim_image)
    print(df_test)
    trained = keras.models.load_model('/gpfs/home/cj3272/56/APPRANTI/cj3272/unet/unet_weights.01-0.2285.hdf5')
    results = trained.predict_generator(test_generator, workers=8, use_multiprocessing=True, verbose=1)
    assert batch_size == len(results)
    import scipy.misc

    my_mask = results[0]
    assert isinstance(my_mask, np.ndarray)
    import PIL.Image as Image

    scipy.misc.imsave('outfile.jpg', Image.fromarray(my_mask[:, :, 0]))

    # EDIT: The current scipy version started to normalize all images so that min(data) become black and max(data) become white.
    # This is unwanted if the data should be exact grey levels or exact RGB channels. The solution:
    # scipy.misc.toimage(image_array, cmin=0.0, cmax=...).save('outfile.jpg')
