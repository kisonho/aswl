# import system modules
from typing import List, Tuple, Optional

# import tensorflow modules
import tensorflow as tf 
from keras import Model, metrics, optimizers
# from keras.layers import Conv2D, Dense

# import core modules 
from aswl.layers import AttentionLayer
from aswl.regularizers import AttentionRegularizer
from aswl.compression import Utils

# compressible model
class CompressibleModel(Model):
    # properties
    attention_regularizer: Optional[AttentionRegularizer]
    metrics: List[metrics.Metric]
    optimizer: optimizers.TFOptimizer
    uncompressed_weights: List[tf.Variable]
    _alpha: float
    _attentions: List[tf.Variable]

    @property
    def attention_layers(self) -> List[AttentionLayer]:
        # initialize layers
        attention_layers: List[AttentionLayer] = list()

        # loop for base model layers
        for l in self.layers: 
            # check layer type
            if isinstance(l, AttentionLayer):
                attention_layers.append(l)
        return attention_layers

    @property
    def attentions(self) -> List[tf.Variable]:
        return self._attentions

    @property
    def filters(self) -> List[int]:
        filters: List[int] = list()

        # get attentions
        for l in self.attention_layers:
            filters.append(l.channels)
        return filters

    # constructor
    def __init__(self, *args, pruning_factor: float=1, attention_regularizer: Optional[AttentionRegularizer]=None, **kwargs):
        super().__init__(*args, **kwargs) 

        # initialize attentions
        self._alpha = pruning_factor
        self._attentions: List[tf.Variable] = list()
        filters: List[int] = list()
        self.attentions_optimizer = None

        # get attentions
        for l in self.attention_layers:
            self._attentions.append(l.attention)
            filters.append(l.channels)
        
        # initialize properties
        self.attention_regularizer = attention_regularizer

        # add regularizer to loss
        if self.attention_regularizer is not None:
            self.attention_regularizer.filters = filters
            self.add_loss(lambda: self.attention_regularizer(self.attentions) if self.attention_regularizer is not None else 0)
        
    # initialize uncompressed weights
    def initialize_uncompressed_weights(self):
        # uncompressed scope
        with tf.name_scope('uncompressed'):
            # initialize uncompressed weights
            self.uncompressed_weights = list()

            # compress base model layers
            for l in self.attention_layers:
                # loop for weights
                for w in l.trainable_variables:
                    uncompressed_weight = tf.Variable(w, trainable=False, shape=w.shape, name=w.name.replace('/', '_').replace(':', '.'))
                    self.uncompressed_weights.append(uncompressed_weight)

    # load uncompressed weights
    def load_uncompressed_weights(self):
        # initialize count
        count: int = 0
        
        # compress base model layers
        for l in self.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                w.assign(self.uncompressed_weights[count])
                count += 1

    # record uncompressed weights
    def record_uncompressed_weights(self):
        # initialize count
        count: int = 0

        # compress base model layers
        for l in self.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                self.uncompressed_weights[count].assign(w)
                count += 1

    # get compressible weights
    def get_compressible_weights(self) -> List[tf.Variable]:
        # initialize weights
        weights: List[tf.Variable] = list()

        # loop for attention layers
        for l in self.attention_layers:
            weights.extend(l.compressible_weights)
        return weights

    # compile model
    def compile(self, *args, optimizer: Optional[optimizers.Optimizer]=optimizers.RMSprop(), apply_attention: bool=False, **kwargs):
        # apply attention
        if apply_attention is True:
            # loop for attention layers
            for l in self.attention_layers:
                l.has_attention_applied = True

        # uncompressed scope
        with tf.name_scope('uncompressed'):
            # initialize uncompressed weights
            for l in self.attention_layers:
                # initialize uncompressed weights
                self.uncompressed_weights = list()

                # compress base model layers
                for l in self.attention_layers:
                    # loop for weights
                    for w in l.trainable_variables:
                        uncompressed_weight = tf.Variable(w, trainable=False, shape=w.shape, name=w.name.replace('/', '_').replace(':', '.'))
                        self.uncompressed_weights.append(uncompressed_weight)

        # regular compile
        return super().compile(*args, optimizer=optimizer, **kwargs) # type: ignore

    # compress
    def compress(self, attentions: tf.Tensor) -> None:
        # initialize counter
        count = 0

        # convert attentions to tensor
        attentions = tf.convert_to_tensor(attentions)
        pruning_ratio = tf.pow((1 - attentions), self._alpha) # type: ignore
        pruning_ratio = tf.clip_by_value(pruning_ratio, 0, 0.99)
        self.pruning_ratios = pruning_ratio
        
        # get attention layers
        i: int = 0
        
        # compress single layer method
        for l in self.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                self.uncompressed_weights[count].assign(w)
                count += 1

            # compress conv2d and dense layer
            Utils.compress(l, pruning_ratio[i]) # type: ignore
            i += 1

    # train one step
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> dict:
        # unzip data
        x_train, y_train = data

        # forward pass
        with tf.GradientTape() as tape:
            y_pred: tf.Tensor = self(x_train, training=True)  # Forward pass
            assert self.compiled_loss is not None and self.compiled_metrics is not None, "[Runtime Error]: Model has not been compiled yet."
            loss: tf.Tensor = self.compiled_loss(y_train, y_pred, regularization_losses=self.losses)
            self.compiled_metrics.update_state(y_train, y_pred)

        # backward pass
        self.load_uncompressed_weights()
        gradients: List[tf.Tensor] = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # zip summary
        summary: dict = dict()
        summary['loss'] = loss
        for m in self.metrics:
            summary[m.name] = m.result()

        # limit attentions between 0.01 and 1
        attentions: tf.Tensor = tf.clip_by_value(self.attentions, 0.01, 1) # type: ignore

        # compress layers
        self.compress(attentions)
        return summary