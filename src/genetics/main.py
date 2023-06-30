from functools import partial
import tensorflow as tf
import random

# класс описывает возможные активационные функции
class Activation:
    activations = ['relu', 'sigmoid', 'tanh']

    @classmethod
    def indexed(cls, arr_str):
        # создаем пустой список
        indexes = []
        # итерируемся по каждому элементу входного массива
        for act_str in arr_str:
            # если элемент есть в списке доступных активаций, то добавляем его индекс в список
            if act_str in cls.activations:
                indexes.append(cls.activations.index(act_str))
        return indexes


from functools import partial
import tensorflow as tf

 
class Layer:
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.params = []

    def addParameter(self, param_name, param_range):
        self.params.append((param_name, param_range))

    def build(self, input_tensor):
        layer_module = getattr(layers, self.layer_name)
        for param_name, param_range in self.params.items():
            random_value = random.uniform(param_range.start, param_range.stop)
            layer_module = partial(layer_module, **{param_name: random_value})
        x = layer_module()(input_tensor)
        return x

# класс wrapper для составления модели keras
class ModelDefiner:
    
    def __init__(self):
        self.inputs = []
        self.layers = []

    def add_input(self, input_tensor):
        layer = Layer(None)
        layer.output_tensor = input_tensor
        self.inputs.append(input_tensor)
        self.layers.append(layer)
        return layer

    def add_layer(self, layer_name):
        layer = Layer(layer_name)
        self.layers.append(layer)
        return layer

    def generate_model(self):
        if len(self.inputs) == 1:
            input_tensor = self.inputs[0]
        else:
            input_tensor = self.inputs
        
        x = input_tensor
        for layer in self.layers:
            if layer.layer_name is not None:
                x = layer.build(x)
        return tf.keras.models.Model(inputs=input_tensor, outputs=x)

# Это генератор массива(гены) бота генетического алгоритма , который использует ModelDefiner для извлечения структуры модели
class bot_generator:
   def __init__(self, model_generator):
     self.model_generator = model_generator
   
   def generate(self):
    # генерация массива бота
    bot = []
    for layer in self.model_generator.layers:
     layer_name = layer.layer_name
   
     for param_name, param_range in layer.params:
        if isinstance(param_range, range):
            bot.append(random.randint(param_range.start, param_range.stop))
        else:
            bot.append(random.choice(param_range))
    return bot
    
   # генерирует модель из model_generator на основе заданных значений в массиве бота
   def generate_model(self, bot_array):
     inputs = self.model_generator.inputs
     outputs = inputs
     for i, layer in enumerate(self.model_generator.layers):
         if bot_array[i] == 1:
             outputs = layer.build(outputs)
     return tf.keras.models.Model(inputs=inputs, outputs=outputs)

# Использование ModelDefiner
model_generator = ModelDefiner()

input_tensor = tf.keras.Input(shape=(32,)) #
input_tensor = tf.keras.layers.Input()
x = tf.keras.layers.Dense()()
x = tf.keras.layers.concatenate(inputs)
# Хочу реализовать по аналогии функционального программирования кераса

input1   = model_generator.add_input(input_tensor)
model = tf.keras.models.Sequential()
model.add(layer)

x = model_generator.add_layer('Flatten')(input1)
x = model_generator.add_layer('Flatten')(x)
#x = model_generator.add_layer('RepeatVector')(x)
#x.addParameter('n', list[3,2])

print(model_generator.layers)
print(bot_generator(model_generator).generate())

model = model_generator.generate_model()
model.summary()