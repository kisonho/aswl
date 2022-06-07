# import tensorflow modules
from typing import Optional, Tuple
import tensorflow as tf

# import core modules
from applications import resnet

#  build resnet50v2
def ResNet50V2(include_top: bool=True, input_shape: Tuple[int, int, int]=(224,224,3), pooling: Optional[str]=None, classes: int=1000, classifier_activation=tf.nn.softmax) -> Tuple[tf.Tensor, tf.Tensor]:
    """Instantiates the ResNet50V2 architecture."""
    def stack_fn(x: tf.Tensor) -> tf.Tensor:
        x = resnet.stack2(x, 64, 3, name='conv2')
        x = resnet.stack2(x, 128, 4, name='conv3')
        x = resnet.stack2(x, 256, 6, name='conv4')
        return resnet.stack2(x, 512, 3, stride1=1, name='conv5')

    return resnet.ResNet(
        stack_fn,
        True,
        True,
        include_top,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation,
    )

