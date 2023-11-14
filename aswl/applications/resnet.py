# import tensorflow modules
import tensorflow as tf
from tensorflow import keras as K

# import core modules
from layers import AttentionConv2D as Conv2D
from layers import AttentionDense as Dense

def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    """A residual block.
    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
            otherwise identity shortcut.
        name: string, block label.
    Returns:
        Output tensor for the residual block.
    """
    bn_axis = 3 if K.backend.image_data_format() == 'channels_last' else 1

    preact = K.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = K.layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(preact)
    else:
        shortcut = K.layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = K.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = K.layers.Activation('relu', name=name + '_1_relu')(x)

    x = K.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = Conv2D(
        filters,
        kernel_size,
        strides=stride,
        use_bias=False,
        name=name + '_2_conv')(x)
    x = K.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = K.layers.Activation('relu', name=name + '_2_relu')(x)

    x = Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = K.layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Arguments:
        x: input tensor.
        filters: integer, filters of the bottleneck layer in a block.
        blocks: integer, blocks in the stacked blocks.
        stride1: default 2, stride of the first layer in the first block.
        name: string, stack label.
    Returns:
        Output tensor for the stacked blocks.
    """
    x = block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = block2(x, filters, name=name + '_block' + str(i))
    x = block2(x, filters, stride=stride1, name=name + '_block' + str(blocks))
    return x

def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           input_shape=(224,224,3),
           pooling=None,
           classes=1000,
           classifier_activation=tf.nn.softmax,
           **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.
    Caution: Be sure to properly pre-process your inputs to the application.
    Please see `applications.resnet.preprocess_input` for an example.
    Arguments:
        stack_fn: a function that returns output tensor for the
        stacked residual blocks.
        preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
        layer at the top of the network.
        weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
        (i.e. output of `K.layers.Input()`)
        to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
        classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        **kwargs: For backwards compatibility only.
    Returns:
        A `keras.Model` instance.
    Raises:
        ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
        ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    img_input = K.layers.Input(shape=input_shape)

    bn_axis = 3 if K.backend.image_data_format() == 'channels_last' else 1

    x = K.layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = K.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = K.layers.Activation('relu', name='conv1_relu')(x)

    x = K.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = K.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = K.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = K.layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = K.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        # imagenet_utils.validate_activation(classifier_activation, weights)
        x = Dense(classes, name='predictions')(x)
        x = classifier_activation(x)
    else:
        if pooling == 'avg':
            x = K.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = K.layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    inputs = img_input
    return inputs, x