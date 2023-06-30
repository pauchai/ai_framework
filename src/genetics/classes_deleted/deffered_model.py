
import tensorflow as tf
from typing import List, Type, TypeVar

class ModelDefiner:
    layers: List[type(DefferedLayer)] = []

    def __init__(self):
        pass

    def defer(self, layer_cls, params = None, vparams = None)->DefferedLayer:
        deffered_layer = DefferedLayer(layer_cls, params, vparams)
        self.layers.append(deffered_layer)
        return deffered_layer
  

T = TypeVar('T', bound='DefferedLayer')
class DefferedLayer:
    inputs: List[T] = None
    params = None
    layer = None
    def __init__(self, layer, params, variableParams):
        self.layer = layer
        # запоминаем параметры
        self.params = params
        self.variableParams = variableParams

    def setInputs(self, inputs: List[T])->T:
        self.inputs = inputs
        return self
    
    def __call__(self, inputs: List[T])->T:
        return self.setInputs(inputs)

T = TypeVar('T', bound='LayerResolver')
class LayerResolver:
    def __init__(self, deferred_layer: DefferedLayer):
        self.deferred_layer = deferred_layer

    def resolve(self):
        result = self.deferred_layer.layer(**self.deferred_layer.params)

        if not self.deferred_layer.inputs  is None:
            inputs = [LayerResolver(deffered_input).resolve() for deffered_input in self.deferred_layer.inputs]
            result = result(inputs)
        return result

class ModelBuilder:
    def __init__(self, modelDefiner):
        self.modelDefiner = modelDefiner
    
    def model(self, inputs, outputs )->tf.keras.models.Model:  
        inputs = [LayerResolver(input_layer).resolve() for input_layer in inputs] 
        outputs = [LayerResolver(output_layer).resolve() for output_layer in outputs]
        return tf.keras.models.Model(inputs, outputs)
