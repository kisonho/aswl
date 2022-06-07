from keras import layers, Model
from keras.models import * # type: ignore

from .layers import AttentionConv2D, AttentionDense
from .models import Model as CompressibleModel

def apply_layer(layer: layers.Layer) -> layers.Layer:
    if isinstance(layer, layers.Conv2D):
        config = layer.get_config()
        return AttentionConv2D(**config)
    elif isinstance(layer, layers.Dense):
        config = layer.get_config()
        return AttentionDense(**config)
    else: return layer

def apply(model: Model, **kwargs) -> CompressibleModel:
    cloned_model: Model = clone_model(model, clone_function=apply_layer)
    return CompressibleModel(inputs=cloned_model.input, outputs=cloned_model.output, name=f"attentioned_{cloned_model.name}", **kwargs)