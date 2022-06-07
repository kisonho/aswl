from typing import Any, Dict

# import tensorflow
import tensorflow as tf 
from keras.metrics import * # type: ignore

# import core modules
from .keras import CompressibleModel as Model

class AveragePruningRatio(Metric):
    _pruning_factor: float
    _model: Model
    _pruning_ratio: tf.Variable
    _gpu_num_reduce: int

    def __init__(self, model: Model, pruning_factor: float, gpu_num_reduce: int=1, **kwargs):
        super().__init__(**kwargs)

        # initialize properties
        self._pruning_factor = pruning_factor
        self._model = model
        self._gpu_num_reduce = gpu_num_reduce

        # initialize pruning ratio
        self._pruning_ratio = self.add_weight(name='avg_pruning_ratio', initializer='zeros', dtype=tf.float64)

    def result(self) -> tf.Tensor:
        return tf.convert_to_tensor(self._pruning_ratio)

    def reset_state(self):
        pass

    def update_state(self, *args, **kwargs) -> tf.Tensor:
        # calculate current pruning ratios
        attentions = tf.convert_to_tensor(self._model.attentions, tf.float64)
        attentions = tf.clip_by_value(attentions, 0.01, 1)
        pruning_ratio = tf.pow(1 - attentions, self._pruning_factor) # type: ignore
        pruning_ratio = tf.clip_by_value(pruning_ratio, 0, 0.99)
        filters = tf.convert_to_tensor(self._model.filters, dtype=tf.int64)

        # evaluate average pruning ratio
        avg_pruning_ratio = tf.cast(pruning_ratio * tf.cast(filters, tf.float64), dtype=tf.int64) # type: ignore
        avg_pruning_ratio = tf.cast(tf.reduce_sum(avg_pruning_ratio), tf.float64) / tf.cast(tf.reduce_sum(filters), tf.float64) # type: ignore
        avg_pruning_ratio = tf.cast(avg_pruning_ratio, tf.float64) / self._gpu_num_reduce # type: ignore
        self._pruning_ratio.assign(avg_pruning_ratio)
        return tf.convert_to_tensor(self._pruning_ratio)