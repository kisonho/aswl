# import system modules
from typing import List

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from layers import Compressing

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
        flatten_weights = Utils.flatten(w)
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

    # dequantization method
    @staticmethod
    def dequantize(quantized_weights: list) -> list:
        # find minimum of the weights
        min_weight, max_weight = -6, 6

        # initialize weights
        weights = list()

        # loop weights
        for w in quantized_weights:
            w = tf.quantization.dequantize(w, min_weight, max_weight)
            weights.append(w)
        return weights
        
    # fake quantize a weight but keep type to be float32
    @staticmethod
    def fake_quantize(weights: list, num_bits: int) -> list:
        '''
        weights - a Tensor of weight
        num_bits - an int for the quantization bits
        returns a Tensor of quantized weight
        '''
        # initialize quantization
        max_weight = float(tf.reduce_max(weights))
        min_weight = float(tf.reduce_min(weights))

        # quantize function
        quantized_weights = tf.quantization.fake_quant_with_min_max_args(weights, max=max_weight, min=min_weight, num_bits=num_bits)
        return quantized_weights
        
    # quantize weight
    @staticmethod
    def quantize(weights: list) -> list:
        # find minimum of the weights
        min_weight, max_weight = -6, 6
        
        # initialize quantized weights
        quantized_weights = list()
        
        # loop weights
        for w in weights:
            w = tf.quantization.quantize(w, min_weight, max_weight, T=tf.qint8)
            quantized_weights.append(w)
        
        # return quantized weight
        return quantized_weights

    # sparse a parameter
    @staticmethod
    def convert_to_sparse(parameter: tf.Tensor) -> tf.SparseTensor:
        '''
        parameter - a Tensor to convert
        returns a tf.sparse.SparseTensor of the given tensor
        '''
        # get indices where value is not 0
        non_zero_condition = tf.not_equal(0, parameter)
        indices = tf.where(non_zero_condition)
        
        # get sparse parameter
        values = tf.gather_nd(parameter, indices)
        sparse_parameter = tf.sparse.SparseTensor(indices, values, parameter.get_shape())
        return sparse_parameter

    # return to dense
    @staticmethod
    def convert_to_dense(sparse_parameter: tf.SparseTensor) -> tf.Tensor:
        '''
        sparse_parameter - a tf.sparse.SparseTensor to convert
        returns a dense Tensor of the given sparse tensor
        '''
        parameter = tf.sparse.to_dense(sparse_parameter)
        return parameter

    # compress a layer
    @staticmethod
    def compress(layer: Compressing, p: float):
        # initialize pruning
        weights = layer.get_compressible_weights()
        magnitude = Utils.calculate_magnitude(weights, p)

        # prune layer
        for w in weights:
            pruned_weight = Utils.prune(w, magnitude)
            # quantized_weight = Utils.quantize(pruned_weight)
            # w.assign(quantized_weight)