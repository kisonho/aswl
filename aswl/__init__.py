from . import compression, keras, layers, metrics, regularizers
from .keras import CompressibleModel as Model

def apply(model: keras.Model, **kwargs) -> Model:
    return Model(inputs=model.input, outputs=model.output, name=f"attentioned_{model.name}", **kwargs)