# import system modules
from typing import Optional, List

# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from models import CompressibleModel
from compression import Utils

# fine tune optimizer for clip-q
class CompressionOptimizer(K.optimizers.Optimizer):
    # properties
    _model: Optional[CompressibleModel]
    _weights_optimizer: K.optimizers.Optimizer
    _attentions_optimizer: Optional[K.optimizers.Optimizer]
    _alpha: tf.Variable
    uncompressed_weights: Optional[List[tf.Variable]]

    @property
    def model(self) -> Optional[CompressibleModel]:
        return self._model

    @property
    def weights_optimizer(self) -> K.optimizers.Optimizer:
        return self._weights_optimizer
    
    @property
    def attentions_optimizer(self) -> K.optimizers.Optimizer:
        return self._attentions_optimizer if self._attentions_optimizer is not None else self.weights_optimizer

    @property
    def alpha(self) -> tf.Variable:
        return self._alpha

    @property
    def lr(self) -> float:
        return self.weights_optimizer.lr

    @lr.setter
    def lr(self, lr: float):
        self.weights_optimizer.lr = lr

    # constructor
    def __init__(self, weights_optimizer: K.optimizers.Optimizer, attentions_optimizer: Optional[K.optimizers.Optimizer]=None, alpha: float=1, name='compression', **kwargs):
        super().__init__(name) # type: ignore

        # initialize variables
        self._model = None
        self._weights_optimizer = weights_optimizer
        self._attentions_optimizer = attentions_optimizer
        
        # attention scope
        with tf.name_scope('attention'):
            self._alpha = tf.Variable(alpha, dtype=tf.float32, trainable=False, name='alpha')

    # initialize uncompressed weights
    def initialize_uncompressed_weights(self, model: CompressibleModel):
        # record model
        self._model = model
        
        # uncompressed scope
        with tf.name_scope('uncompressed'):
            # initialize uncompressed weights
            self.uncompressed_weights = list()

            # compress base model layers
            for l in model.attention_layers:
                # loop for weights
                for w in l.trainable_variables:
                    uncompressed_weight = tf.Variable(w, trainable=False, name=w.name.replace('/', '_').replace(':', '.'))
                    self.uncompressed_weights.append(uncompressed_weight)

    # compress
    def compress(self, pruning_ratio: List[float], *args, **kwargs):        
        # initialize layer index and pruning ratio
        layer_index = 0
        
        # initialize counter
        count = 0

        # compress base model layers
        for l in self.model.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                self.uncompressed_weights[count].assign(w)
                count += 1

            # compress conv2d and dense layer
            Utils.compress(l, pruning_ratio[layer_index])

    # reset weights
    def reset_weights(self, *args, **kwargs):
        # initialize recorded weights count
        count = 0

        # compress base model layers
        for l in self.model.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                w.assign(self.uncompressed_weights[count])
                count += 1

    # get gradients
    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        # unzip grads and vars
        [gradients, variables] = list(zip(*grads_and_vars))

        # initialize variables and gradients
        attentions_vars: List[tf.Variable] = list()
        attentions_grads: List[tf.Tensor] = list()
        weights_vars: List[tf.Variable] = list()
        weights_grads: List[tf.Variable] = list()

        # initialize gradients count
        gradients_count: int = 0

        # get attention gradients and attentions
        for v in variables:
            # check variable name
            if "attention" in str(v.name).split('/')[-1]:
                attentions_vars.append(v)
                attentions_grads.append(gradients[gradients_count]) 
            else:
                weights_vars.append(v)
                weights_grads.append(gradients[gradients_count])

            # implement gradients count
            gradients_count += 1

        # zip grads and vars
        attentions_grads_and_vars = zip(attentions_grads, attentions_vars)
        weights_grads_and_vars = zip(weights_grads, weights_vars)

        # reset uncompressed weights
        self.reset_weights()

        # update procedure
        self.weights_optimizer.apply_gradients(weights_grads_and_vars, *args, **kwargs)
        self.attentions_optimizer.apply_gradients(attentions_grads_and_vars, *args, **kwargs)

        # limit attention
        attentions: tf.Tensor = tf.convert_to_tensor(attentions_vars)
        attentions = tf.clip_by_value(attentions, 0.01, 1)

        # convert to pruning ratios
        pruning_ratio = tf.pow((1 - attentions), self._alpha)

        # compress
        self.compress(pruning_ratio)

    # get config
    def get_config(self) -> dict:
        return self.weights_optimizer.get_config()