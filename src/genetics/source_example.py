#get_pred() – обученная модель. Предсказывает результат, который возвращается к ненормированным данным;
def get_pred(currModel,
             xVal,
             yVal,
             yScaler
             ):
  '''
  Функция расчетов прогнозирования сети.

  -----------------
  Входные данные:
  currModel - нейронная сеть;
  xVal - x тестовая выборка;
  yVal - y тестовая выборка;
  yScaler - скейлер данных.
  -----------------
  На выходе функции:
  predVal - результаты предсказания;
  yValUnscaled - правильные ответы в исходной размерности
  (какими они были до нормирования).

  '''
  # Вычисление и деномализация предсказания
  predVal = yScaler.inverse_transform(currModel.predict(xVal))
  # Денормализация верных ответов
  yValUnscaled = yScaler.inverse_transform(yVal)
  # И возвращаем исходны масштаб данных, до нормализации
  return (predVal, yValUnscaled)

#correlate() – расчет коэффициента автокорреляции;
# Функция расёта корреляции дух одномерных векторов
def correlate(a,
              b
              ):
  '''
  Функция расчета корреляции между двумя списками.
    
  -----------------
  Входные данные:
  a - первый вектор;
  b - второй вектор;
  -----------------
  На выходе функции:
  val - значение корреляции.
  '''

  # Рассчитываем основные показатели
  ma = a.mean() # Среднее значение первого вектора
  mb = b.mean() # Среднее значение второго вектора
  mab = (a*b).mean() # Среднее значение произведения векторов
  sa = a.std() # Среднеквадратичное отклонение первого вектора
  sb = b.std() # Среднеквадратичное отклонение второго вектора
  
  #Рассчитываем корреляцию
  val = 0
  if ((sa>0) & (sb>0)):
    val = (mab-ma*mb)/(sa*sb)
  return val
  # аналог функции в нампи np.corrcoef(a, b)[0, 1]

#show_predict() – построение графиков предсказания и верных ответов;
def show_predict(start,
                 step,
                 channel,
                 predVal,
                 yValUnscaled
                 ):
  '''
  Функция визуализирует графики, что предсказала сеть
  и какие были правильные ответы.  

  -----------------
  Входные данные:
  start - точка с которой начинаем отрисовку графика
  step - длина графика, которую отрисовываем
  channel - какой канал отрисовываем
  predVal - результаты предсказания;
  yValUnscaled - правильные ответы в исходной размерности
  -----------------
  На выходе функции:
  Функция визуализирует графики.
  '''
  plt.figure(figsize=(12, 5))
  plt.plot(predVal[start:start+step, channel], 
           label='Прогноз')
  plt.plot(yValUnscaled[start:start+step, channel], 
           label='Базовый ряд')
  plt.xlabel('Время')
  plt.ylabel('Значение Close')
  plt.legend()
  plt.show()
  
#auto_corr() – расчет и построение графика автокорреляции;
def auto_corr(channels,
              corrSteps,
              predVal,
              yValUnscaled,
              plot_graf = True,
              return_data = False):
  '''
  Функция расчета корреляции и автокорреляции.
  А также может визуализировать результаты в графики.
    
  -----------------
  Входные данные:
  channels - по каким каналам отображать корреляцию;
  corrSteps - количество шагов смещения назад для рассчёта корреляции;
  predVal - результаты предсказания;
  yValUnscaled - правильные ответы в исходной размерности;
  plot_graf - рисовать или нет графики (по умолчанию True);
  return_data - выводить или нет данные (по умолчанию False).
  -----------------
  На выходе функции:
  Если plot_graf = True функция рисует корреляцию спрогнозированного сигнала
  с исходным, смещая на различное количество шагов назад
  Если return_data = True функция выводит значения:
  corr - корреляции по шагам исходных знчений с предсказанными;
  own_corr - корреляции исходных знчений с самими собой
  '''
  # Проходим по всем каналам
  for ch in channels:
    corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно
    yLen = yValUnscaled.shape[0] # Запоминаем размер проверочной выборки

      # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
      # Получаем сигнал, смещённый на i шагов назад
      # predVal[i:, ch]
      # Сравниваем его с верными ответами, без смещения назад
      # yValUnscaled[:yLen-i,ch]
      # Рассчитываем их корреляцию и добавляем в лист
      corr.append(correlate(yValUnscaled[:yLen-i,ch], predVal[i:, ch]))

    own_corr = [] # Создаём пустой лист, в нём будут корреляции при смезении на i рагов обратно

      # Постепенно увеличикаем шаг, насколько смещаем сигнал для проверки автокорреляции
    for i in range(corrSteps):
      # Получаем сигнал, смещённый на i шагов назад
      # predVal[i:, ch]
      # Сравниваем его с верными ответами, без смещения назад
      # yValUnscaled[:yLen-i,ch]
      # Рассчитываем их корреляцию и добавляем в лист
      own_corr.append(correlate(yValUnscaled[:yLen-i,ch], yValUnscaled[i:, ch]))

    # Отображаем график коррелций для данного шага
    if plot_graf: #Если нужно показать график
      plt.figure(figsize=(12, 5))
      plt.plot(corr, label='предсказание на ' + str(ch+1) + ' шаг')
      plt.plot(own_corr, label='Эталон')

  if plot_graf: #Если нужно показать график
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.show()

  if return_data: #Если нужно вернуть массивы автокорреляции
     return corr, own_corr

# Создаём сеть (net - список параметров)
def create_randnet(net,
                   xLen,
                   channels,
                   num_cls,
                   control_level_shape = 10**4):
  '''
  Функция формирования нелинейной нейронной ссети из бота.
  
  -----------------
  Входные данные:
  net - полученный бот со списком значений для формирования сети
  xLen - размер анализируемых данных до предсказания;
  channels - количество каналов данных;
  num_cls - количество предсказываемых шагов;
  control_level_shape - парамметр при превышению которого будет применен 
  слой GlobalAveragePooling1D, а не Flatten (по умолчанию 10'000).
  -----------------
  На выходе функции:
  model - нелинейная нейронная сеть Model()
  '''
  # определяем форму входных данных
  input_shape = (xLen, channels)

  '''
  Входной блок
  '''
  makeFirstNormalization = net[0] # Делаем ли нормализацию в начале
  FirstDenseorConv = net[1]       # Тип входного блока Dense или Conv
  firstSize = 2 ** net[2]         # размер cвёрточного слоя или к-во нейронов
  firstConvKernel = net[3]        # Ядро для водного свёрточного слоя
  activation0 = net[4]            # Функция активации входного слоя


  '''
  Данные для внутренних блоков
  '''
  maxPoolKernel = net[5]          # Ядро пуллинга для всех Polling слоев
  dropoutRate =  net[6]           # размер дропаута для всех Dropout слоев
  # формировние типа веток
  qtyLstmways = net[7]            # количество веток lstm 
  qtyDenseways = net[8]           # количество веток  dense
  qtyConvways = net[9]            # количество веток  conv


  '''
  Первый скрытый блок
  '''
  makeway1 = net[10]               # Делаем ли ветку
  waySize1 = 2 ** net[11]          # размер слоя (LSTM, Dense, Conv)
  wayKernel1 = net[12]             # Размер Kernel если свёрточная ветка
  waysriides1 = net[13]           # Sriides если свёрточная ветка
  waypadding1 = net[14]           # Рadding если свёрточная ветка
  activation1 = net[15]           # Функция активации ветки
  wayPoolDrop1 = net[16]          # Делаем ли пуллинг|дропаут соответственно

  '''
  Второй скрытый блок
  '''
  makeway2 = net[17]               # Делаем ли ветку
  waySize2 = 2 ** net[18]          # размер слоя (LSTM, Dense, Conv)
  wayKernel2 = net[19]             # Размер Kernel если свёрточная ветка
  waysriides2 = net[20]            # Sriides если свёрточная ветка
  waypadding2 = net[21]            # Рadding если свёрточная ветка
  activation2 = net[22]            # Функция активации ветки
  wayPoolDrop2 = net[23]           # Делаем ли пуллинг|дропаут соответственно

  '''
  Третий скрытый блок
  '''
  makeway3 = net[24]               # Делаем ли ветку
  waySize3 = 2 ** net[25]          # размер слоя (LSTM, Dense, Conv)
  wayKernel3 = net[26]             # Размер Kernel если свёрточная ветка
  waysriides3 = net[27]            # Sriides если свёрточная ветка
  waypadding3 = net[28]            # Рadding если свёрточная ветка
  activation3 = net[29]            # Функция активации ветки
  wayPoolDrop3 = net[30]           # Делаем ли пуллинг|дропаут соответственно

  '''
  Четвертый скрытый блок
  '''
  makeway4 = net[31]               # Делаем ли ветку
  waySize4 = 2 ** net[32]          # размер слоя (LSTM, Dense, Conv)
  wayKernel4 = net[33]             # Размер Kernel если свёрточная ветка
  waysriides4 = net[34]            # Sriides если свёрточная ветка
  waypadding4 = net[35]            # Рadding если свёрточная ветка
  activation4 = net[36]            # Функция активации ветки
  wayPoolDrop4 = net[37]           # Делаем ли пуллинг|дропаут соответственно

  '''
  Пятый скрытый блок
  '''
  makeway5 = net[38]               # Делаем ли ветку
  waySize5 = 2 ** net[39]          # размер слоя (LSTM, Dense, Conv)
  wayKernel5 = net[40]             # Размер Kernel если свёрточная ветка
  waysriides5 = net[41]            # Sriides если свёрточная ветка
  waypadding5 = net[42]            # Рadding если свёрточная ветка
  activation5 = net[43]            # Функция активации ветки
  wayPoolDrop5 = net[44]           # Делаем ли пуллинг|дропаут соответственно

  '''
  Выходной dense блок
  '''
  makeDense = net[45]              # Делаем ли препоследний полносвязный
  denseSize = 2 ** net[46]         # Размер полносвязного слоя
  activation6 = net[47]            # Фукнция активации пятго слоя

  '''
  Список активационных функций
  '''
  activation_list = ['linear','relu', 'elu', 'selu' ,'tanh','softmax','sigmoid'] 

  '''
  Условия для формирования блоков
  _______________________________
  '''

  '''
  Входной блок Dense или Conv
  '''
  # Входной слой
  inputs = Input(input_shape)  

  # Если делаем нормализацию в начале
  if (makeFirstNormalization):    
    x = BatchNormalization()(inputs)

    if (FirstDenseorConv):
        x = Dense(firstSize, activation=activation_list[activation0])(x)
        x = Activation(activation_list[activation0])(x)
    else:
        x = Conv1D(firstSize,firstConvKernel, padding ='same')(x)
        x = Activation(activation_list[activation0])(x)
  # Если не делаем нормализацию в начале
  else:                           
    if (FirstDenseorConv):
        x = Dense(firstSize, activation=activation_list[activation0])(inputs)
        x = Activation(activation_list[activation0])(x)
    else:
        x = Conv1D(firstSize,firstConvKernel, padding ='same')(inputs)
        x = Activation(activation_list[activation0])(x)

  # Список для сборы выходов всех веток включая выход из inputs
  list_to_concat = [inputs]       


  '''
  ОПРЕДЕЛЕНИЕ ТИПА ВЕТОК
  '''
  gensway = 7 # количество генов в ветке
  # счетчик индексов
  idxgen = lambda idx0way, gensway, i: idx0way+gensway*i 
  
  
  '''
  Ветка слоев на основе Lstm
  '''
  idx0way = 10 # нулевой ген ветки Lst
  #print('qtyLstmways', net[7])
  if qtyLstmways: # если колиство веток Lstm больше 0
      #print('LSTM: ', idx0way)
      # Проходимся по каждому блоку
      for i in range(qtyLstmways):              
          idx0lay = idxgen(idx0way,gensway,i)
          #print('LSTM - ', idx0lay)
          if net[idx0lay]!=0: # Добавление блока
              x = LSTM(max(3, net[idx0lay+1]), return_sequences=True)(x)
              x = Activation(activation_list[net[idx0lay+5]])(x)
              x = Reshape((-1,1))(x)
              list_to_concat.append(x)  # Иначе сразу добавляем в список

  '''
  Ветка слоев на основе Dense
  '''
  idx0way += qtyLstmways*gensway # нулевой ген ветки Dense   
  #print('qtyDenseways',net[8])
  if qtyDenseways: # если колиство веток Dense больше 0
    # Проходимся по каждому блоку
      #print('DENSE: ', idx0way)
      for i in range(qtyDenseways):              
          idx0lay = idxgen(idx0way,gensway,i)
          #print('DENSE - ', idx0lay)
          if net[idx0lay]!=0: # Добавление блока
              x = Dense(net[idx0lay+1])(x)
              x = Activation(activation_list[net[idx0lay+5]])(x)
              if net[idx0lay+6]!=0:           # Добавление пулинга
                x = Dropout(dropoutRate)(x)
                x = Reshape((-1,1))(x)
                list_to_concat.append(x)  # Добавляем в список

              else: # Иначе сразу добавляем в список
                x = Reshape((-1,1))(x)
                list_to_concat.append(x)  

  '''
  Ветка слоев на основе Conv
  '''
  #print('qtyConvways' , net[9])
  idx0way += qtyDenseways*gensway # нулевой ген ветки Conv     
  if qtyConvways: # если колиство веток Conv больше 0
      # Проходимся по каждому блоку
      #print('CONV: ', idx0way)   
      for i in range(qtyConvways):              
          idx0lay = idxgen(idx0way,gensway,i)
          #print('CONV - ', idx0lay)  
          if net[idx0lay]!=0: # Добавление блока
              x = Conv1D(net[idx0lay+1], net[idx0lay+2],
                         strides = net[idx0lay+3],
                         padding = net[idx0lay+4])(x)
              x = Activation(activation_list[net[idx0lay+5]])(x)

              if net[idx0lay+6]!=0:           # Добавление пулинга
                x = MaxPooling1D(maxPoolKernel)(x)
                x = Reshape((-1,1))(x)
                list_to_concat.append(x)  # Добавляем в список
              else:   # Иначе сразу добавляем в список
                x = Reshape((-1,1))(x)
                list_to_concat.append(x)  

      
          list_to_concat.append(x)    # Добавляем в список

  '''
  Блок проверки размерности слоя
  Для применения Flatten() или GlobalAveragePooling1D()
  для вытягивания в вектор значений и передачи в выходной блок из Dense слоев
  '''
  # Проходим по всем значениям списка list_to_concat и делаем flatten
  for i in range(len(list_to_concat)):    
    if list_to_concat[i].shape != (None,0,1):
          # Получаем размерность последнего из добавленных слоев
        control_shape = list_to_concat[i].get_shape()

        if control_shape[-1]*control_shape[-2] < control_level_shape:
          # Добавляем слой Flatten
            list_to_concat[i] = Flatten()(list_to_concat[i])                           
        else:
            # Добавляем слой GlobalAveragePooling1D
            list_to_concat[i] = GlobalAveragePooling1D()(list_to_concat[i])             
    else:
      break

  '''
  Блок соединения веток и вытягивания в вектор
  '''
  # Соединяем значения списка в единое целое
  if len(list_to_concat) != 1:
     fin = concatenate(list_to_concat)
  
  # Иначе просто делаем flatten
  else:
    fin = x
    fin = Flatten()(fin)

  '''
  Выходной dense блок
  '''
  # Добавление полносвязного слоя
  if makeDense!=0:
    fin = Dense(denseSize)(fin)
    fin = Activation(activation_list[activation6])(fin)

  # выходной слой
  fin = Dense(num_cls)(fin)

  model = Model(inputs, fin)  # Создаем модель 
  return model                # Возвращаем моель

def create_bot4net():
  '''
  Функция создания списка случайных параметров - бота

  -----------------
  Входные данные:
  отсутствуют
  -----------------
  На выходе функции:
  net - список параметров для линейной нс
  '''
  # количество внутренних веток, которые описаны нижн
  qtyways = 5

  '''
  Гены входного блока
  '''
  net = []
  net.append(random.randint(0,1))  # 0 Делаем или нет нормализацию
  net.append(random.randint(0,1))  # 1 Тип входного блока Dense или Conv
  net.append(random.randint(3,10)) # 2 размер cвёрточного слоя или к-во нейронов
  net.append(random.randint(2,7))  # 3 Ядро для водного свёрточного слоя
  net.append(random.randint(0,6))  # 4 Функция активации входного слоя

  '''
  Гены парамметров типов внутренних веток будут защищены от изменний
  '''
  net.append(random.randint(2,4))           # 5 Ядро пуллинга веток
  net.append(random.choice((0.1,0.2,0.3)))  # 6 размер дропаута веток
  
  # формировние типа веток
  qtyLstmways = random.randint(0,1)         # 7 количество веток lstm
  net.append(qtyLstmways)  # добавляем количество веток lstm

  qtyDenseways = random.randint(0,qtyways-qtyLstmways) # 8 количество веток dense
  net.append(qtyDenseways)# добавляем количество веток dense

  qtyConvways = qtyways - qtyLstmways - qtyDenseways # 9 количество веток conv
  net.append(qtyConvways) # добавляем количество веток conv

  '''
  Гены для первой ветки
  '''
  net.append(random.randint(0,1))  # 10 Делаем ли 1ю ветку
  net.append(random.randint(3,10)) # 11 размер слоя (LSTM, Dense, Conv) от 8 до 1024
  net.append(random.randint(2,7))  # 12 Размер Kernel если свёрточная ветка от 2 до 7
  net.append(random.randint(1,5))  # 13 Размер Strides если свёрточная ветка от 1 до 5
  net.append(random.choice(('valid','same'))) # 14 pading ветоки
  net.append(random.randint(0,6))  # 15 Функция активации ветки
  net.append(random.randint(0,1))  # 16 Делаем ли пуллинг|дропаут соответственно

  '''
  Гены для второй ветки
  '''
  net.append(random.randint(0,1))  # 17 Делаем ли 2ю ветку
  net.append(random.randint(3,10)) # 18 размер слоя (LSTM, Dense, Conv) от 8 до 1024
  net.append(random.randint(2,7))  # 19 Размер Kernel если свёрточная ветка от 2 до 7
  net.append(random.randint(1,5))  # 20 Размер Strides если свёрточная ветка от 1 до 5
  net.append(random.choice(('valid','same'))) # 21 pading ветоки
  net.append(random.randint(0,6))  # 22 Функция активации ветки
  net.append(random.randint(0,1))  # 23 Делаем ли пуллинг|дропаут соответственно

  '''
  Гены для третьей ветки
  '''
  net.append(random.randint(0,1))  # 24 Делаем ли 3ю ветку
  net.append(random.randint(3,10)) # 25 размер слоя (LSTM, Dense, Conv) от 8 до 1024
  net.append(random.randint(2,7))  # 26 Размер Kernel если свёрточная ветка от 2 до 7
  net.append(random.randint(1,5))  # 27 Размер Strides если свёрточная ветка от 1 до 5 
  net.append(random.choice(('valid','same'))) # 28 pading ветоки
  net.append(random.randint(0,6))  # 29 Функция активации ветки
  net.append(random.randint(0,1))  # 31 Делаем ли пуллинг|дропаут соответственно

  '''
  Гены для четвертой ветки
  '''
  net.append(random.randint(0,1))  # 31 Делаем ли 4ю ветку
  net.append(random.randint(3,10)) # 32 размер слоя (LSTM, Dense, Conv) от 8 до 1024
  net.append(random.randint(2,7))  # 33 Размер Kernel если свёрточная ветка от 2 до 7
  net.append(random.randint(1,5))  # 34 Размер Strides если свёрточная ветка от 1 до 5
  net.append(random.choice(('valid','same'))) # 35 pading ветоки
  net.append(random.randint(0,6))  # 36 Функция активации ветки
  net.append(random.randint(0,1))  # 37 Делаем ли пуллинг|дропаут соответственно

  '''
  Гены для пятоой ветки
  '''
  net.append(random.randint(0,1))  # 38 Делаем ли 5ю ветку
  net.append(random.randint(3,10)) # 39 размер слоя (LSTM, Dense, Conv) от 8 до 1024
  net.append(random.randint(2,7))  # 40 Размер Kernel если свёрточная ветка от 2 до 7
  net.append(random.randint(1,5))  # 41 Размер Strides если свёрточная ветка от 1 до 5
  net.append(random.choice(('valid','same'))) # 42 pading ветоки
  net.append(random.randint(0,6))  # 43 Функция активации ветки
  net.append(random.randint(0,1))  # 44 Делаем ли пуллинг|дропаут соответственно

  '''
  Гены для выходного блока
  '''
  net.append(random.randint(0,1))  # 45 Делаем ли дополнительный полносвязный слой
  net.append(random.randint(3,10)) # 46 Размер полносвязного слоя от 8 до 1024
  net.append(random.randint(0,6))  # 47 Функция активации
  return net

bot = create_bot4net()
print(bot)
print('Длина бота', len(bot))
# Создаем модель createConvNet
model = create_randnet(bot,
                      inputShape[1], # количество подаваемых шагов в наборе
                      inputShape[2], # количество каналов данных в наборе
                      outputShape[1] # на сколько предсказываем 
                      )
# выводим слои модели 
#model.summary()  

#функция оценки бота
def eval_net(net,
             ep,
             verb,
             xData,
             yData,
             xLen,
             channels,
             num_cls,
             x_test,
             y_test,
             Scaler
             ):
  '''
  Функция вычисления результатов работы сети:
  
  -----------------
  Входные данные:
  net - бот популяции;
  ep - к-во эпох проверки нс;
  verb - выводить или нет процесс обучения нс на эпохе;
  xData - тренировочные данные;
  yData - проверочные данные;
  xLen - количество подаваемых шагов в наборе;
  channels - количество каналов данных в наборе;
  num_cls - на сколько предсказываем;
  x_test - x тестовая выборка;
  y_test - y тестовая выборка;
  Scaler - скейлер данных;
  ------------------------
  На выходе функции:
  val - на выходе функции оценка работы нс на эпохах проверки
  '''
  val = 0
  model = create_randnet(net, xLen,  channels, num_cls) # Создаем модель createConvNet
  
  # Компилируем модель
  model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='mse')
  print('Обучение модели бота', net)
  history = model.fit(xData,
                    epochs=ep, 
                    verbose=verb,
                    validation_data=yData)
    

  # Прогнозируем данные текущей сетью
  y_pred, y_true = get_pred(model, x_test, y_test, Scaler) #Прогнозируем данные

  print('Сохранение оценки бота') 
  # Возвращаем автокорреляцию 
  corr, own_corr = auto_corr([0], 3, y_pred, y_true, plot_graf = False, return_data = True)
  val = MAE(corr, own_corr).numpy()*history.history["val_loss"][-1]# Считаем MAE и прибавляем ошибку
  
  return val                      # Возвращаем точность

#применение 
eval_net(bot,               # бот популяции 
            3,              # к-во эпох проверки 
            1,              # выводить или нет процесс обучения 
            train_datagen,  # тренировочные данные
            val_datagen,    # проверочные данные
            inputShape[1],  # количество подаваемых шагов в наборе
            inputShape[2],  # количество каналов данных в наборе
            y_test.shape[1], # на сколько предсказываем 
            x_test,         # тестовая выборка
            y_test,         # тестовая выборка
            y_scaler        # скейлер данных
            )

#отбор ботов
def search():
    '''
    Основные параметры для поиска
    '''
    n = 15              # Общее число ботов
    nsurv = 5           # Количество выживших (столько лучших переходит в новую популяцию)
    newrand = 5
    nnew = n - nsurv - newrand    # Количество новых (столько новых ботов создается)
    l = 48              # Размер бота
    epohs = 20          # количество эпох поиска
    control_std = 0.0003 # выход из поиска если изменение в лучших ниже данного значения

    '''
    Особенности мутации
    '''
    mut = 0.5        # стартовый коэфициент мутаций
    eph_change_mut = (2, 4, 6, 8) # эпохи смены коэфициента мутации
    new_mut = (0.4, 0.3, 0.2, 0.1) # новый коэфициент мутаци

    '''
    Защищенные гены
    '''
    notchangeidx = (7, 8, 9) # индексы бота защищенные и скрещивания от мутации
    notmutidx = (5, 6) # индексы бота защищенные от мутации


    '''
    Создаём популяцию случайных ботов
    '''
    popul = []         # Массив популяции
    val = []           # Одномерный массив значений этих ботов
    for i in range(n):
        popul.append(create_bot4net())


    '''
    Основной цикл поиска
    '''  
    sval_best = []    # Одномерный массив значений лучших ботов на эпохах
    # Пробегаем по всем эпохам
    for it in range(epohs):                 
        # проверяем текущую эпоху it на принадлежность графику смены мутации
        if it in eph_change_mut:
            idx = eph_change_mut.index(it) # получаем индекс  по эпохе
            mut = new_mut[idx] # проверяем текущую эпохуобновляем мутацию
            print('Смена мутации на', mut)
            print()  

        val = []                              # Обнуляем значения бота
        curr_time = time.time()               # засекаем время

        '''
        Получение оценок ботов
        '''  
        # Пробегаем в цикле по всем ботам 
        for i in range(n):                    
            bot = popul[i]                     # Берем очередного бота

            # Вычисляем точность текущего бота
            f = eval_net(bot,           # бот популяции 
                        3,              # к-во эпох проверки 
                        0,              # выводить или нет процесс обучения 
                        train_datagen,  # тренировочные данные
                        val_datagen,    # проверочные данные
                        inputShape[1],  # количество подаваемых шагов в наборе
                        inputShape[2],  # количество каналов данных в наборе
                        outputShape[1], # на сколько предсказываем 
                        x_test,         # тестовая выборка
                        y_test,         # тестовая выборка
                        y_scaler        # скейлер данных
                        ) 
            val.append(f)   # Добавляем полученное значение в список val
        
        '''
        Сортировка оценок ботов и контроль поиса
        ''' 
        sval = sorted(val, reverse=0)         # Сортируем val
        # Выводим 5 лучших ботов
        print(it, time.time() - curr_time, " ", sval[0:5],popul[:5]) 

        sval_best.append(sval[0])             # добавляем значение лучшего бота
        # проверка на продолжение поиска, есть разница или уже нет в точности
        if it > 5:                            # с 6й эпохи 
            sval_best = sorted(sval_best, reverse=0)[:5] # сортируем и берем 5ть лучших 
            if np.std(sval_best) < control_std:          # сверяем значения на отличие 
                print('Поиск дучших не дает нового, выход')
                break

        '''
        Сохранение лучших ботов в newpopul
        '''  
        newpopul = [] # Создаем пустой список под новую популяцию
        # Пробегаем по всем выжившим ботам
        for i in range(nsurv):
            # Получаем индекс очередного бота из списка лучших в списке val             
            index = val.index(sval[i])
            # Добавляем в новую популяцию бота из popul с индексом index        
            newpopul.append(popul[index])       
        '''
        Создание новых ботов на основе лучших ботов в newpopul.
        Иногда дополнительно применение мутации и исключения!
        '''
        # Проходимся в цикле nnew-раз 
        for i in range(nnew):
            # случайный выбор родителя в диапазоне от 0 до nsurv - 1              
            indexp1 = random.randint(0,nsurv-1) # Случайный индекс 1го родителя 
            indexp2 = random.randint(0,nsurv-1) # Случайный индекс 1го родителя
            botp1 = newpopul[indexp1]           # бота-родителя 1 по indexp1
            botp2 = newpopul[indexp2]           # бота-родителя  2 по indexp2    
            newbot = []                         # пустой список для нового бота    
            net4Mut = create_bot4net()         # Создаем случайную сеть для мутаций

            # выбираем основного родителя для защищенных генов
            randparent = random.choice((botp1,botp2))

            '''
            Пробегаем по всем генам бота
            '''
            for j in range(l): 
                # Если ген незащищен от скрещивания 
                if j not in notchangeidx:                       
                    x = 0      
                    '''
                    Скрещивание
                    '''
                    # Получаем случайное число в диапазоне от 0 до 1
                    pindex = random.random() 
                    # Если pindex меньше 0.5, то берем значения от 1 бота, иначе от 2
                    if pindex < 0.5: x = botp1[j]
                    else: x = botp2[j]
                    
                    '''
                    Мутация
                    '''
                    # Если ген незащищен от мутации
                    if j not in notmutidx:
                        # С вероятностью mut устанавливаем значение бота из net4Mut
                        if (random.random() < mut): x = net4Mut[j]
                        else: pass

                    # Если ген защищен берем от основного         
                    else: x = randparent[j]
                # Если ген защищен берем от осноаного
                else: x = randparent[j]
                newbot.append(x)    # Добавляем очередное значение в нового бота      
            newpopul.append(newbot) # Добавляем бота в новую популяцию      
            '''
            Добавление случайных ботов для разнообразия
            ''' 
            for i in range(newrand):
                newpopul.append(create_bot4net())
        popul = newpopul   
        