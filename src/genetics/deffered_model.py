import tensorflow as tf

class BotDefiner:
    gens:List[Func]

    def addGen(self, randFunc):
        self.gens.append(randFunc)
    def size(self):
        return len(self.gens)
    
    '''
    generate rand bot
    '''
    def generate(self):
        return [f() for f in gens]
            
class DefferedLayer:
    inputs = None
    params = None
    var_params = None
    layer = None
    def __init__(self, layer, params: dict = None):
        self.layer = layer
        # запоминаем параметры
        self.params = params
    def setInputs(self, inputs):
        self.inputs = inputs
        return self

    '''
    @var_param_name 
    @randFunc ex: lambda: random.randint(0,1)
    '''
    def addVariableParams(self, var_param_name, randFunc):
        botDefiner.addGen(randFunc)
        self.var_params.append({var_param_name:botDefiner.size - 1}) # {param_name: bot_idx}
        

    def __call__(self, inputs):
        return self.setInputs(inputs)


class ModelDefiner:

    layers = None
    def __init__(self):
        self.layers = []

    def defer(self, layer: tf.keras.layers.Layer, params: dict = None) -> DefferedLayer:
        deffered_layer = DefferedLayer(layer, params, inputs)
        self.layers.append(deffered_layer)
        return deffered_layer



class RandIfOperation:
    op:Operation
    def __init__(self, op, range_num = 3):
        botDefiner.addGen(lambda : random.randint(0,range_num))
        bot_idx = botDefiner.size - 1

    def __call__(self):
        if (self.randFunc()):
            self.op()
        else:

        

class RandRangeOperation:
    randfunc:func

    
