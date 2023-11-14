# import system modules
from typing import Any, Optional, List
from typing_extensions import Protocol, runtime_checkable

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# compression protocol
@runtime_checkable
class Compressing(Protocol):
    # get weights
    def get_compressible_weights(self) -> List[tf.Variable]:
        return NotImplemented

# attentioning protocol
class Attentioning(Protocol):
    # properties
    @property
    def channels(self) -> int:
        return self._get_channels()

    # apply attention to weights
    def apply_attention(self):
        return NotImplemented

    # clear attentions
    def clear_attention(self):
        return NotImplemented

    # protected get channels
    def _get_channels(self) -> int:
        return NotImplemented

# attention based conv2d
class AttentionLayer(K.layers.Layer, Attentioning, Compressing):
    # properties
    _attention: tf.Variable
    _attention_layer: K.layers.Layer

    @property
    def attention(self) -> tf.Variable:
        return self._attention

    @attention.setter
    def attention(self, attention: tf.Variable):
        self._attention = attention

    @property
    def attention_layer(self) -> K.layers.Layer:
        return self._attention_layer

    # constructor
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize attention
        self.attention = tf.Variable(0.5, dtype=tf.float32, trainable=True, name='attention')

    # apply attention
    def apply_attention(self):
        # get weights
        weights = self._attention_layer.trainable_variables

        # apply attention to weights
        for w in weights:
            attentioned_weight = w * self.attention
            w.assign(attentioned_weight)

        # clear attention
        self.clear_attention()

    # build layer
    def build(self, input_shape: tuple):
        self._attention_layer.build(input_shape)

    # call layer
    def call(self, input_data: Any) -> tf.Tensor:
        # call attention layer
        y: tf.Tensor = self._attention_layer(input_data)

        # attention multiplier
        y = y * self.attention
        return y

    # get compressible weights
    def get_compressible_weights(self) -> List[tf.Variable]:
        return self._attention_layer.trainable_variables

    # clear attentions
    def clear_attention(self):
        self.attention.assign(1)

    # get config
    def get_config(self):
        return self._attention_layer.get_config()

    # protected get channels
    def _get_channels(self) -> int:
        return 0

# attention based conv2d
class AttentionConv2D(AttentionLayer):
    # properties alias
    _attention_layer: K.layers.Conv2D

    # constructor
    def __init__(self, *args, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)

        # initialize attention layer
        self._attention_layer = K.layers.Conv2D(*args, name=name, **kwargs)

    # get channels
    def _get_channels(self) -> tf.Tensor:
        kernel_shape: tf.TensorShape = self._attention_layer.kernel.shape
        channels: tf.Tensor = tf.reduce_prod(kernel_shape)
        return channels

# attention based dense
class AttentionDense(AttentionLayer):
    # properties alias
    _attention_layer: K.layers.Dense

    # constructor
    def __init__(self, *args, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)

        # initialize attention layer
        self._attention_layer = K.layers.Dense(*args, name=name, **kwargs)

    # get channels
    def _get_channels(self) -> int:
        kernel_shape: tf.TensorShape = self._attention_layer.kernel.shape
        channels: tf.Tensor = tf.reduce_prod(kernel_shape)
        return channels
