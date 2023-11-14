# import tensorflow
import tensorflow as tf 
from tensorflow import keras as K

# import core modules
from models import CompressibleModel

# convert attentions to pruning ratios
def convert_attentions_to_pruning_ratios(attentions: tf.Tensor, filters: tf.Tensor, beta: float, is_average_ratio: bool=True) -> tf.Tensor:
    avg_pruning_ratio: tf.Tensor = attentions * filters
    avg_pruning_ratio = tf.reduce_sum(avg_pruning_ratio) / tf.reduce_sum(filters) if is_average_ratio is True else avg_pruning_ratio / filters
    avg_pruning_ratio = tf.pow(1 - avg_pruning_ratio, beta)
    return avg_pruning_ratio

# average pruning ratio metrics
class AveragePruningRatio(K.metrics.Metric):
    # properties
    _pruning_factor: float
    _model: CompressibleModel
    _pruning_ratio: tf.Variable

    # constructor
    def __init__(self, model: CompressibleModel, pruning_factor: float, name: str="average_pruning_ratio"):
        super().__init__(name=name, dtype=tf.float32)

        # initialize properties
        self._pruning_factor = pruning_factor
        self._model = model
        self._pruning_ratio = self.add_weight(name='avg_pruning_ratio', initializer='zeros')

    # get result
    def result(self) -> tf.Tensor:
        return self._pruning_ratio

    # reset state
    def reset_states(self):
        pass

    # update state
    @tf.function
    def update_state(self, *args, **kwargs):
        # calculate current pruning ratios
        attentions: tf.Tensor = tf.convert_to_tensor(self._model.attentions, tf.float32)
        filters: tf.Tensor = tf.convert_to_tensor(self._model.filters, tf.float32)
        avg_pruning_ratio: tf.Tensor = convert_attentions_to_pruning_ratios(attentions, filters, self._pruning_factor, False)
        self._pruning_ratio.assign(avg_pruning_ratio)
        return tf.convert_to_tensor(self._pruning_ratio)