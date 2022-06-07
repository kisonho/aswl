# import system modules
from typing import Any, Optional, List

# import tensorflow modules
import tensorflow as tf
from keras.layers import * # type: ignore

class AttentionLayer(Layer):
    _attention: tf.Variable
    _attention_layer: Layer
    _has_attention_applied: bool

    @property
    def attention(self) -> tf.Variable:
        return self._attention

    @attention.setter
    def attention(self, attention: tf.Variable):
        self._attention = attention

    @property
    def attention_layer(self) -> Layer:
        return self._attention_layer

    @property
    def channels(self) -> int: return 0

    @property
    def compressible_weights(self) -> List[tf.Variable]:
        return [getattr(self._attention_layer, "kernel")]

    @property
    def has_attention_applied(self) -> bool:
        return self._has_attention_applied

    @has_attention_applied.setter
    def has_attention_applied(self, has_attention_applied: bool):
        self._has_attention_applied = has_attention_applied

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize attention
        self.attention = tf.Variable(0.5, dtype=tf.float32, trainable=True, aggregation=tf.VariableAggregation.MEAN, name='attention')
        self._has_attention_applied = False

    def build(self, input_shape: tuple):
        self._attention_layer.build(input_shape)

    def call(self, input_data: Any) -> tf.Tensor:
        # call attention layer
        y: tf.Tensor = self._attention_layer(input_data)

        # attention multiplier
        if self._has_attention_applied is False:
            attention = tf.clip_by_value(self.attention, 0.01, 1)
            y = y * self.attention # type: ignore
        return y

class AttentionConv2D(AttentionLayer):
    # properties alias
    _attention_layer: Conv2D

    @property
    def channels(self) -> tf.Tensor:
        return tf.reduce_prod(self._attention_layer.kernel.shape)

    # constructor
    def __init__(self, *args, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)

        # initialize attention layer
        self._attention_layer = Conv2D(*args, name=f"attentioned_{name}", **kwargs)

# attention based dense
class AttentionDense(AttentionLayer):
    # properties alias
    _attention_layer: Dense
    
    @property
    def channels(self) -> tf.Tensor:
        return tf.reduce_prod(self._attention_layer.kernel.shape)

    # constructor
    def __init__(self, *args, name: Optional[str]=None, **kwargs):
        super().__init__(name=name)

        # initialize attention layer
        self._attention_layer = Dense(*args, name=f"attentioned_{name}", **kwargs)
