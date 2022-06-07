# import system modules
from typing import List, Protocol, Union, runtime_checkable

# import tensorflow modules
import tensorflow as tf
from keras.layers import Layer

# compression protocol
@runtime_checkable
class Compressing(Protocol):
    @property
    def attention(self) -> tf.Variable:
        return NotImplemented

    @property
    def compressible_weights(self) -> List[tf.Variable]:
        return NotImplemented
    
    @property
    def attention_layer(self) -> Layer:
        return NotImplemented

# compression utils
class Utils(object):
    # flatten a list
    @staticmethod
    def flatten(list_to_flatten: List[tf.Variable]) -> tf.Tensor:
        # flatten weights
        flatten_list = list()

        # loop for each item
        for l in list_to_flatten:
            l = tf.reshape(l, (-1,))
            flatten_list.append(l)
        
        # return
        return tf.concat(flatten_list, axis=0)

    # calculate magnitude
    @staticmethod
    def calculate_magnitude(w: List[tf.Variable], p: float) -> float:
        # sort weights
        flatten_weights = tf.reshape(w, (-1,))
        abs_flatten_weights = tf.abs(flatten_weights)
        sorted_weights = tf.sort(flatten_weights)

        # calculate magnitude
        magnitude_index = tf.cast(abs_flatten_weights.shape[0] * p, tf.int32)
        magnitude_index = tf.minimum(magnitude_index, abs_flatten_weights.shape[0] - 1)
        return sorted_weights[magnitude_index]

    # prune method
    @staticmethod
    def prune(w: tf.Variable, m: float) -> tf.Tensor:
        '''
        parameters:
            w - weights
            m - magnitude
        '''
        # calculate mask
        abs_weights = tf.abs(w)
        pruned_mask = tf.cast(tf.greater_equal(abs_weights, m), tf.float32)

        # apply mask
        pruned_weights = w * pruned_mask
        return pruned_weights

    # compress a layer
    @staticmethod
    def compress(layer: Compressing, p: float):
        # initialize pruning
        weights = layer.compressible_weights

        # prune layer
        for w in weights:
            # pruned weight
            magnitude = Utils.calculate_magnitude(w, p)
            w = Utils.prune(w, magnitude)

    # remove compression wrap
    @staticmethod
    def remove(layer: Union[Layer, Compressing]) -> Layer:
        if isinstance(layer, Compressing):
            for w in layer.compressible_weights:
                w.assign(layer.attention * w)
            return layer.attention_layer
        else: return layer
