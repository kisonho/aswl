# import system modules
from typing import List

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# attention regularizer
class AttentionRegularizer(K.regularizers.Regularizer):
    # properties
    _r: float
    _alpha: float
    _attention_decay: float
    _filters: List[int]

    @property
    def r(self) -> float:
        return self._r

    @property
    def attention_decay(self) -> float:
        return self._attention_decay

    @property
    def filters(self) -> List[int]:
        return self._filters

    @filters.setter
    def filters(self, f: List[int]):
        self._filters = f

    # constructor
    def __init__(self, attention_decay: float=1, pruning_factor: float=1, r: float=0):
        super().__init__()
        
        # initialize properties
        self._r = r
        self._alpha = pruning_factor
        self._attention_decay = attention_decay

    # call method
    @tf.function
    def __call__(self, attentions: List[tf.Variable]) -> float:
        # calculate pruning ratio
        pruning_ratio: tf.Tensor = tf.convert_to_tensor(attentions, dtype=tf.float32)
        pruning_ratio = tf.pow(1 - pruning_ratio, self._alpha)

        # S = (1 - p) * nW / sum(nW)
        penalty: tf.Tensor = (1 - pruning_ratio) * tf.convert_to_tensor(self.filters, dtype=tf.float32)
        penalty = tf.reduce_sum(penalty) / tf.cast(tf.reduce_sum(self.filters), tf.float32)

        # penalty = decay * S^2
        penalty -= self._r
        penalty = tf.math.square(penalty)
        penalty = self._attention_decay * penalty
        return penalty

# attention
def attention(decay: float=0.5, alpha: float=1, r: float=1) -> AttentionRegularizer:
    return AttentionRegularizer(decay, alpha, r)
