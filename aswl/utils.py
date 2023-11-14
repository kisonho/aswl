# import system modules
from typing import List, Tuple, Optional

# import tensorflow
import tensorflow as tf
from tensorflow import keras as K
# class data preprocessor
class ImagePreprocessor(object):
    # properties
    _input_shape: Tuple[int, int, int]
    _is_crop_enabled: bool

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self._input_shape

    @property
    def is_crop_enabled(self) -> bool:
        return self._is_crop_enabled

    # constructor
    def __init__(self, input_shape: Tuple[int, int, int], is_crop_enabled: bool=True):
        self._input_shape = input_shape
        self._is_crop_enabled = is_crop_enabled
        
    # preprocessing imagenet v2 dataset
    def preprocess(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # cast images to float
        image = tf.cast(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        
        # crop
        if self._is_crop_enabled is True:
            image = self.scale_image(image)
            image = tf.image.random_crop(image, self._input_shape)

        # preprocess
        image = K.applications.mobilenet_v2.preprocess_input(image)
            
        # set shape
        image.set_shape(self.input_shape)
        return image, label
    
    # scale image
    def scale_image(self, image: tf.Tensor) -> tf.Tensor:
        # get image shape
        image_shape: tf.TensorShape = tf.shape(image)
        size: tf.Tensor = tf.random.uniform(shape=[], minval=256, maxval=480, dtype=tf.int32)
        size = tf.cast(size, tf.float32)
        
        # define larger width function
        def calculate_larger_width_size() -> tf.Tensor:
            ratio: float = image_shape[1] / image_shape[0]
            h: tf.Tensor = tf.cast(size, tf.int32)
            w: tf.Tensor = tf.cast(tf.cast(ratio, tf.float32) * size, dtype=tf.int32)
            return tf.image.resize(image, (h, w))
            
        # define larger height function
        def calculate_larger_height_size() -> tf.Tensor:
            ratio: float = image_shape[0] / image_shape[1]
            w: tf.Tensor = tf.cast(size, tf.int32)
            h: tf.Tensor = tf.cast(tf.cast(ratio, tf.float32) * size, dtype=tf.int32)
            return tf.image.resize(image, (h, w))
            
        # scale image
        image = tf.cond(image_shape[0] <= image_shape[1], true_fn=calculate_larger_width_size, false_fn=calculate_larger_height_size)
        return image

