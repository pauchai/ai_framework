"""

необходимо протестировать класс ModelDefiner имеющий метод defer ,
которая принимает в качестве аргумента tensorflow.keras.layers.Layer
и словарь параметров. внутри этого метода возвращается 
объект DefferedLayer , 
Нужно сделать первый тест внутри которого нужно сождать
1. Mock интерфейса класса ModelDefiner
2. Имплементировать этот класс 
3. и выполнить тест на проверку метода defer(tf.keras.layers.Input, params{'shape': (3,)})

#input_layer = md.defer(tf.keras.layers.Input, params = {'shape':(3,)} ) 

"""
import unittest
from unittest import mock
import tensorflow as tf
from genetics.deffered_model import ModelDefiner, DefferedLayer

class ModelDefinerTestCase(unittest.TestCase):
    def setUp(self):
        self.model_definer = ModelDefiner()
    def tearDown(self):
        self.model_definer = None

    def test_defer(self):
        layer = tf.keras.layers.Input
        params = {'shape': (3,)}
        result = self.model_definer.defer(layer, params)
        
        #definer.defer.assert_called_once_with(input_layer, params)
        self.assertIsInstance(result, DefferedLayer)
        self.assertEqual(result.params, params)
        self.assertIsNone(result.inputs)
        self.assertEqual(result.layer, layer)
        self.assertEqual(len(self.model_definer.layers), 1)

    def test_defer_with_output(self):
        input_layer =  tf.keras.layers.Input
        input_params = {'shape': (3,)} 
        input_deffer_layer = DefferedLayer(
            layer = input_layer,
            params = input_params
        )

        layer = tf.keras.layers.Dense
        params = {'units': 10}
        result = self.model_definer.defer(layer, params)
        result = result([input_deffer_layer])
        
        #definer.defer.assert_called_once_with(input_layer, params)
        self.assertIsInstance(result, DefferedLayer)
        self.assertEqual(result.params, params)
        self.assertEqual(result.layer, layer)
        self.assertIsNotNone(result.inputs)
        self.assertEqual(result.inputs, [input_deffer_layer])

if __name__ == '__main__':
    unittest.main()
