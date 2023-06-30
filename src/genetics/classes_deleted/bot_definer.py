from classes.model_definer import ModelDefiner
class BotDefiner:

    model_definer: ModelDefiner
    
    def __init__(self, model_definer: ModelDefiner):
       
        self.model_definer = model_definer
    
class BotBuilder:
    bot_definer: BotDefiner
    genom: List
    def __init__(self, bot_definer:BotDefiner):
       
        self.bot_definer = bot_definer

    def create_genom(self):
        for defer_layer in model_definer.layers:
            pass # требует реализации


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
