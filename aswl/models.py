# import system modules
from typing import List, Tuple, Callable, Optional, Any
import threading, time

# import tensorflow modules
import tensorflow as tf 
from tensorflow import keras as K
# from tensorflow.keras.layers import Conv2D, Dense

# import core modules 
from regularizers import AttentionRegularizer
from layers import AttentionConv2D as Conv2D
from layers import Attentioning, Compressing, AttentionLayer
from layers import AttentionDense as Dense
import applications.resnetv2 as resnetv2

# compressible model
class CompressibleModel(K.Model):
    # properties
    attention_regularizer: Optional[K.regularizers.Regularizer]
    _attentions: List[tf.Variable]
    _uncompressed_weights: List[tf.Variable]

    @property
    def attention_layers(self) -> List[AttentionLayer]:
        # initialize layers
        layers: List[K.layers.Layer] = list()

        # loop for base model layers
        for l in self.layers: 
            # check layer type
            if isinstance(l, AttentionLayer):
                layers.append(l)
        return layers

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
    def __init__(self, *args, attention_regularizer: Optional[K.regularizers.Regularizer]=None, **kwargs):
        super().__init__(*args, **kwargs)        
        # initialize attentions
        self._attentions: List[tf.Variable] = list()
        filters: List[tf.Tensor] = list()

        # get attentions
        for l in self.attention_layers:
            self._attentions.append(l.attention)
            filters.append(l.channels)

        # initialize properties
        attention_regularizer.filters = filters
        self.attention_regularizer = attention_regularizer

        # add regularizer to loss
        if self.attention_regularizer is not None:
            self.add_loss(lambda: self.attention_regularizer(self.attentions))
        
    # initialize uncompressed weights
    def initialize_uncompressed_weights(self):
        # uncompressed scope
        with tf.name_scope('uncompressed'):
            # initialize uncompressed weights
            self._uncompressed_weights = list()

            # compress base model layers
            for l in self.attention_layers:
                # loop for weights
                for w in l.trainable_variables:
                    uncompressed_weight = tf.Variable(w, trainable=False, shape=w.shape, name=w.name.replace('/', '_').replace(':', '.'))
                    self._uncompressed_weights.append(uncompressed_weight)

    # load uncompressed weights
    def load_uncompressed_weights(self):
        # initialize count
        count: int = 0
        
        # compress base model layers
        for l in self.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                w.assign(self._uncompressed_weights[count])

    # record uncompressed weights
    def record_uncompressed_weights(self):
        # initialize count
        count: int = 0

        # compress base model layers
        for l in self.attention_layers:
            # loop for weights
            for w in l.trainable_variables:
                self._uncompressed_weights[count].assign(w)

    # apply attentions
    def apply_attentions(self):
        # loop for layers
        for l in self.attention_layers:
            l.apply_attention()

    # clear attentions
    def clear_attentions(self):
        # loop for layers
        for l in self.attention_layers:
            l.clear_attention()

    # get compressible weights
    def get_compressible_weights(self) -> List[tf.Variable]:
        # initialize weights
        weights: List[tf.Variable] = list()

        # loop for attention layers
        for l in self.attention_layers:
            weights.extend(l.get_compressible_weights())
        return weights

    # compile model
    def compile(self, optimizer='rmsprop', loss: Optional[K.losses.Loss]=None, metrics: Optional[List[K.metrics.Metric]]=None, loss_weights=None, weighted_metrics=None, run_eagerly=None, apply_attention: bool=False, **kwargs):
        # apply attention
        if apply_attention is True:
            # loop for attention layers
            for l in self.attention_layers:
                l.has_attention_applied = True

        # regular compile
        return super().compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights, weighted_metrics=weighted_metrics, run_eagerly=run_eagerly, **kwargs)

    # load weights
    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        # load weights
        super().load_weights(filepath, by_name=by_name, skip_mismatch=skip_mismatch, options=options)

        # apply attention
        for l in self.attention_layers:
            if l.has_attention_applied is True:
                l.apply_attention()

# model builder
class ModelBuilder(object):
    # properties
    input_shape: Tuple[int, int, int]
    num_classes: int
    model: str
    weight_decay: float
    attention_decay: float
    pruning_factor: float

    # constructor
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int, model: str, weight_decay: float=0, attention_decay: float=1, pruning_factor: float=1):
        super().__init__()

        # initialize properties
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = model
        self.weight_decay = weight_decay
        self.attention_decay = attention_decay
        self.pruning_factor = pruning_factor

    # build model
    def build(self) -> CompressibleModel:
        # initialize layers
        build_model: Callable = getattr(self, "build_" + self.model.lower())

        # initialize attention regularizer
        if self.attention_decay != 0:
            attention_regularizer = AttentionRegularizer(self.attention_decay, self.pruning_factor)
        else:
            attention_regularizer = None

        # initialize compressible sequential
        model: CompressibleModel = build_model(attention_regularizer=attention_regularizer)
        return model

    # build vgg16
    def build_vgg16(self, attention_regularizer: Optional[AttentionRegularizer]=None) -> CompressibleModel:
        # input layer
        img_input: tf.Tensor = K.Input(shape=self.input_shape)
        x: tf.Tensor = img_input

        # Block 1
        x = Conv2D(64, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block1_conv1')(x)
        x = K.layers.BatchNormalization(name='block1_batchnorm1')(x)
        x = tf.nn.relu(x, name='block1_relu1')
        x = Conv2D(64, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block1_conv2')(x)
        x = K.layers.BatchNormalization(name='block1_batchnorm2')(x)
        x = tf.nn.relu(x, name='block1_relu2')
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block2_conv1')(x)
        x = K.layers.BatchNormalization(name='block2_batchnorm1')(x)
        x = tf.nn.relu(x, name='block2_relu1')
        x = Conv2D(128, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block2_conv2')(x)
        x = K.layers.BatchNormalization(name='block2_batchnorm2')(x)
        x = tf.nn.relu(x, name='block2_relu2')
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block3_conv1')(x)
        x = K.layers.BatchNormalization(name='block3_batchnorm1')(x)
        x = tf.nn.relu(x, name='block3_relu1')
        x = Conv2D(256, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block3_conv2')(x)
        x = K.layers.BatchNormalization(name='block3_batchnorm2')(x)
        x = tf.nn.relu(x, name='block3_relu2')
        x = Conv2D(256, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block3_conv3')(x)
        x = K.layers.BatchNormalization(name='block3_batchnorm3')(x)
        x = tf.nn.relu(x, name='block3_relu3')
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block4_conv1')(x)
        x = K.layers.BatchNormalization(name='block4_batchnorm1')(x)
        x = tf.nn.relu(x, name='block4_relu1')
        x = Conv2D(512, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block4_conv2')(x)
        x = K.layers.BatchNormalization(name='block4_batchnorm2')(x)
        x = tf.nn.relu(x, name='block4_relu2')
        x = Conv2D(512, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block4_conv3')(x)
        x = K.layers.BatchNormalization(name='block4_batchnorm3')(x)
        x = tf.nn.relu(x, name='block4_relu3')
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block5_conv1')(x)
        x = K.layers.BatchNormalization(name='block5_batchnorm1')(x)
        x = tf.nn.relu(x, name='block5_relu1')
        x = Conv2D(512, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block5_conv2')(x)
        x = K.layers.BatchNormalization(name='block5_batchnorm2')(x)
        x = tf.nn.relu(x, name='block5_relu2')
        x = Conv2D(512, (3, 3), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='block5_conv3')(x)
        x = K.layers.BatchNormalization(name='block5_batchnorm3')(x)
        x = tf.nn.relu(x, name='block5_relu3')
        x = K.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        # flatten layers
        x = K.layers.Flatten()(x)

        # fc6 - batchnorm6 - relu6 - drop6
        x = Dense(4096, kernel_regularizer=K.regularizers.l2(self.weight_decay), name='fc6')(x)
        x = K.layers.BatchNormalization(name='batchnorm6')(x)
        x = tf.nn.relu(x)
        # x = K.layers.Dropout(0.5, name='drop6')(x)

        # fc6 - batchnorm6 - relu6 - drop6
        x = Dense(4096, kernel_regularizer=K.regularizers.l2(self.weight_decay), name='fc7')(x)
        x = K.layers.BatchNormalization(name='batchnorm7')(x)
        x = tf.nn.relu(x)
        # x = K.layers.Dropout(0.5, name='drop7')(x)
        
        # classification layer
        x = Dense(self.num_classes, kernel_regularizer=K.regularizers.l2(self.weight_decay), name='classification')(x)
        y: tf.Tensor = tf.nn.softmax(x)

        # initialize base model
        model: CompressibleModel = CompressibleModel(inputs=img_input, outputs=y, attention_regularizer=attention_regularizer, name='VGG16')
        return model

    def build_resnet56(self, attention_regularizer: Optional[AttentionRegularizer]=None) -> CompressibleModel:
        # build resnet layer function
        def resnet_layer(inputs: tf.Tensor, num_filters: int=16, kernel_size: int=3, strides: int=1, activation: Optional[Callable]=tf.nn.relu, use_batch_normalization: bool=True, is_conv_first: bool=True):
            # define conv layer
            conv_layer = Conv2D(num_filters, kernel_size=kernel_size, kernel_regularizer=K.regularizers.l2(self.weight_decay), strides=strides, padding='same')

            x = inputs
            if is_conv_first is True:
                x = conv_layer(x)
                if use_batch_normalization is True:
                    x = K.layers.BatchNormalization()(x)
                if activation is not None:
                    x = activation(x)
            else:
                if use_batch_normalization is True:
                    x = K.layers.BatchNormalization()(x)
                if activation is not None:
                    x = activation(x)
                x = conv_layer(x)
            return x
            
        # Start model definition.
        num_filters_in: int = 16
        num_res_blocks: int = 6
        num_filters_out: int = 0

        inputs = K.Input(shape=self.input_shape)
        # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
        x = resnet_layer(inputs=inputs, num_filters=num_filters_in, is_conv_first=True)

        # Instantiate the stack of residual units
        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = tf.nn.relu
                use_batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:  # first layer and first stage
                        activation = None
                        use_batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:  # first layer but not first stage
                        strides = 2    # downsample

                # bottleneck residual unit
                y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, activation=activation, use_batch_normalization=use_batch_normalization, is_conv_first=False)
                y = resnet_layer(inputs=y, num_filters=num_filters_in, is_conv_first=False)
                y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, is_conv_first=False)
                
                # first block add
                if res_block == 0:
                    # linear projection residual shortcut connection to match
                    # changed dims
                    x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides, activation=None, use_batch_normalization=False)
                    
                # add layer
                x = x + y

            # update input filters
            num_filters_in = num_filters_out 

        # Add classifier on top.
        # v2 has BN-ReLU before Pooling
        x = K.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = K.layers.AveragePooling2D(pool_size=7)(x)
        x = K.layers.Flatten()(x)
        x = Dense(self.num_classes)(x)
        outputs = tf.nn.softmax(x) 

        # Instantiate model.
        base_model = CompressibleModel(inputs=inputs, outputs=outputs, attention_regularizer=attention_regularizer, name='ResNet56')
        return base_model
        
    # build resnet50 v2
    def build_resnet50v2(self, attention_regularizer: Optional[AttentionRegularizer]=None) -> CompressibleModel:
        # initialize base model for imagenet
        x, y = resnetv2.ResNet50V2(input_shape=self.input_shape, classes=self.num_classes)
        base_model: CompressibleModel = CompressibleModel(inputs=x, outputs=y, attention_regularizer=attention_regularizer, name='ResNet50V2')
        base_model.summary()
        return base_model

    # build mobilenet v2
    def build_mobilenetv2(self, attention_regularizer: Optional[AttentionRegularizer]=None) -> CompressibleModel:
        # input layer
        img_input: tf.Tensor = K.Input(shape=self.input_shape)
        x: tf.Tensor = img_input

        # build residual block
        def build_residual_block(block_inputs: tf.Tensor, output_dim: int, expansion_ratio: int=1, strides: Tuple[int, int]=(1,1), padding: str='same', repeat: int=1, is_shortcut: bool=True, name: str=None) -> tf.Tensor:         
            # initialize inputs
            block: tf.Tensor = block_inputs  

            # repeat blocks
            for i in range(0,repeat):
                # name with index
                name_with_index = name + "_block" + str(i+1)
                
                # initialize bottleneck dim
                input_dim: int = block.shape[-1]
                bottleneck_dim = round(expansion_ratio * input_dim)

                # residual block
                with tf.name_scope(name_with_index):
                    # pw1
                    block = Conv2D(bottleneck_dim, (1,1), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='SAME', name=name_with_index + '_pw1_conv')(block)
                    block = K.layers.BatchNormalization(name=name_with_index + '_pw1_batchnorm')(block)
                    block = tf.nn.relu(block, name=name_with_index + '_pw1_relu')

                    # dw2
                    block = K.layers.DepthwiseConv2D(kernel_size=(3,3), kernel_regularizer=K.regularizers.l2(self.weight_decay), strides=strides, padding=padding, name=name_with_index + '_dw2_conv')(block)
                    block = K.layers.BatchNormalization(name=name_with_index + '_dw2_batchnorm')(block)
                    block = tf.nn.relu(block, name=name_with_index + '_dw2_relu')

                    # pw3
                    block = Conv2D(output_dim, (1,1), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='SAME', name=name_with_index + '_pw3_conv')(block)
                    block = K.layers.BatchNormalization(name=name_with_index + '_pw3_batchnorm')(block)

                    # add layer
                    if is_shortcut is True and strides[0] == 1 and input_dim == output_dim:
                        # extra pointwise layer to output_dim when not equal
                        block = block + block_inputs

                # reset parameters
                strides = (1, 1)
                block_inputs = block
            return block

        # conv1
        x = Conv2D(32, (3,3), strides=(2,2), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='conv1')(x)
        x = K.layers.BatchNormalization(name='batchnorm1')(x)
        x = K.layers.ReLU(name='relu1')(x)

        # residual block 2
        x = build_residual_block(x, 16, name='bn2')
        x = build_residual_block(x, 24, expansion_ratio=6, strides=(2, 2), repeat=2, name='bn3')
        x = build_residual_block(x, 32, expansion_ratio=6, strides=(2, 2), repeat=3, name='bn4')
        x = build_residual_block(x, 64, expansion_ratio=6, strides=(2, 2), repeat=4, name='bn5')
        x = build_residual_block(x, 96, expansion_ratio=6, repeat=3, name='bn6')
        x = build_residual_block(x, 160, expansion_ratio=6, strides=(2, 2), repeat=3, name='bn7')
        x = build_residual_block(x, 320, expansion_ratio=6, name='bn8')

        # pw9
        x = Conv2D(1280, (1,1), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='pw9')(x)
        x = K.layers.BatchNormalization(name='batchnorm9')(x)
        x = tf.nn.relu(x, name='relu9')

        # average pool 10
        x = tf.keras.layers.AveragePooling2D(x.shape[1:-1], (1,1), name='avgpool10')(x)

        # classification
        x = Conv2D(self.num_classes, kernel_size=(1,1), kernel_regularizer=K.regularizers.l2(self.weight_decay), padding='same', name='classification')(x)
        x = tf.squeeze(x, [1,2], name='squeeze')
        y: tf.Tensor = tf.nn.softmax(x)

        # base model 
        base_model: CompressibleModel = CompressibleModel(inputs=img_input, outputs=y, attention_regularizer=attention_regularizer, name='MobileNetV2')
        return base_model