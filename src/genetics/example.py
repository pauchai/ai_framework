import tensorflow as tf
import random


activation_choice = lambda x: random.choice(['relu', 'sigmoid'])
units_choice = lambda x: 2**random.randint(3,10)
kernel_choice = lambda x: random.randint(3,7)
dropout_rate_choice = lambda x: random.choice((0.1, 0.2, 0.3))
strides_choice = lambda x: random.randint(1,3)
padding_choice = lambda x: random.choice(('same'))


md = ModelDefiner()

input_layer = md.defer(tf.keras.layers.Input, params = {'shape':(3,)} ) 

x = md.sequention(
    [
        md.randIf(md.defer(tf.keras.layers.BatchNormalization)),
        md.randIf(
            md.defer(tf.keras.layers.Dense)
                        .addVariableParam('units', units_choice)
        ).randElse(
            md.defer(tf.keras.layers.Conv1D, params = {'padding': 'same'})
                        .addvariableParam('filters', units_choice)
                        .addVariableParam('kernel_size', kernel_choice)
        ),
       
        md.defer(tf.keras.layers.Activation).addVariableParam('activation', activation_choice),
        
        md.randMultipleSelect(

            md.randIf(md.range(md.randIf(md.sequention(
                [
                    md.defer(tf.keras.layers.LSTM, params={'return_sequences':True}).addVariableParam('units', units_choice),
                    md.defer(tf.keras.layers.Activation).addVariableParam('activation', activation_choice),
                    md.defer(tf.keras.layers.Reshape, params={'target_shape': (-1, 1)}).onAddOutput(lambda output: list_to_concat.append(output))
                ],
            )), 2)),
            
            md.randIf(md.range(md.randIf(md.sequention(
                [
                    md.defer(tf.keras.layers.Dense).addVariableParams('units', units_choice),
                    md.defer(tf.keras.layers.Activation).addVariableParams('activation', activation_choice),    
                    md.randIf(
                        md.defer(tf.keras.layers.Dropout).addVariableParams('rate', dropout_rate_choice)
                    ),
                    md.defer(tf.keras.layers.Reshape, params = {'target_shape': (-1,1)}).onAddOutput(lambda output: list_to_concat.append(output))
                ]
            )),2)),
            
            md.randIf(md.range(md.randIf(md.sequention(
                [
                    md.defer(tf.keras.layers.Conv1D)
                        .addVariableParam("filters", units_choice)
                        .addVariableParam("kernel_size", kernel_choice)
                        .addVariableParam("strides", strides_choice)
                        .addVariableParam("padding", padding_choice),
                    md.randIf(
                        md.defer(tf.keras.layers.MaxPooling1D).addVariableParams('pool_size', pool_size_choice)

                    ),
                    md.defer(tf.keras.layers.Activation).addVariableParam('activation', activation_choice),                    
                    md.defer(tf.keras.layers.Reshape, params = {'target_shape': (-1,1)}).onAddOutput(lambda output: list_to_concat.append(output))
                    
                ]
            )),2))
            # list_to_concat.append(x)    # Добавляем в список
            
        )
    ]    
)(input_layer)  


'''
  Блок проверки размерности слоя
  Для применения Flatten() или GlobalAveragePooling1D()
  для вытягивания в вектор значений и передачи в выходной блок из Dense слоев
'''
for i in range(len(list_to_concat)):    
    if list_to_concat[i].shape != (None,0,1):
          # Получаем размерность последнего из добавленных слоев
        control_shape = list_to_concat[i].get_shape()

        if control_shape[-1]*control_shape[-2] < control_level_shape:
          # Добавляем слой Flatten
            list_to_concat[i] = md.defer(tf.keras.layers.Flatten)(list_to_concat[i])                           
        else:
            # Добавляем слой GlobalAveragePooling1D
            list_to_concat[i] = md.defer(tf.keras.layers.GlobalAveragePooling1D)(list_to_concat[i])             
    else:
      break

'''
Блок соединения веток и вытягивания в вектор
'''
# Соединяем значения списка в единое целое
if len(list_to_concat) != 1:
    fin = md.defer(tf.keras.layers.Concatenate)(list_to_concat)

# Иначе просто делаем flatten
else:
    fin = x
    fin = md.defer(tf.keras.layers.Flatten)(fin)

'''
  Выходной dense блок
 '''
# Добавление полносвязного слоя
if makeDense!=0:
    fin = md.defer(tf.keras.layers.Dense).addVariableParams('units', units_choice)(fin)
    fin = md.defer(tf.keras.layers.Activation).addVariableParams('activation', activation_choice)(fin)

# выходной слой
fin = md.defer(tf.keras.layers.Dense, params = {'units': num_cls})(fin)

model_bulder = ModelBuilder(md)
model = md.model(inputs, fin)  # Создаем модель 
#return model                # Возвращаем моель


'''
case_defer_layer(layer,params,inputs)
    *layer
    *params
    *input

case_randIf(case, inputs)
    case = case

case_sequention([case1, case2, case3], inputs)
    cases = []

case_randRange(case, from, to, inputs)
    case = case
    from
    to

'''


'''
другой вариант синтаксиса
'''


md = modelDefiner

input = Input()

md.randIf(
    lambda md:
x = md.defer(tf.keras.layers.Conv1D)
                        .addVariableParam("filters", units_choice)
                        .addVariableParam("kernel_size", kernel_choice)
                        .addVariableParam("strides", strides_choice)
                        .addVariableParam("padding", padding_choice)(input)
    return x
}
