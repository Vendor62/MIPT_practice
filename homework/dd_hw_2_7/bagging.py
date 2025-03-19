import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags  # Количество бэгов (подвыборок)
        self.oob = oob  # Флаг использования OOB-оценки
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Генерация индексов для каждого бэга и сохранение их в self.indices_list
        '''
        self.indices_list = []
        data_length = len(data)
        for _ in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)  # Бустрэп-выборка
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Обучение модели на каждом бэге.
        
        model_constructor — класс модели, передается без параметров и без скобок.
        Пример:
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'Все бэги должны быть одинаковой длины!'
        assert list(map(len, self.indices_list))[0] == len(data), 'Все бэги должны содержать len(data) элементов!'
        
        self.models_list = []
        for indices in self.indices_list:
            model = model_constructor()
            data_bag, target_bag = data[indices], target[indices]  # Создание обучающего набора для данного бэга
            self.models_list.append(model.fit(data_bag, target_bag))  # Обучение и сохранение модели
        
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Получение среднего предсказания для каждого объекта из переданного набора данных
        '''
        predictions = np.array([model.predict(data) for model in self.models_list])
        return np.mean(predictions, axis=0)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Генерация списка списков, где список i содержит предсказания для объекта self.data[i]
        от всех моделей, которые не видели этот объект во время обучения
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        
        for model, indices in zip(self.models_list, self.indices_list):
            oob_mask = np.ones(len(self.data), dtype=bool)
            oob_mask[indices] = False  # Отмечаем индексы, использованные в обучении
            oob_predictions = model.predict(self.data[oob_mask])  # Предсказания для OOB-набора
            
            for i, pred in zip(np.where(oob_mask)[0], oob_predictions):
                list_of_predictions_lists[i].append(pred)
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Вычисление среднего предсказания для каждого объекта обучающего набора.
        Если объект использовался во всех бэгах, возвращается None вместо предсказания.
        '''
        self._get_oob_predictions_from_every_model()
        
        self.oob_predictions = np.array([
            np.mean(preds) if len(preds) > 0 else None for preds in self.list_of_predictions_lists
        ])
    
    def OOB_score(self):
        '''
        Вычисление среднеквадратичной ошибки (MSE) для всех объектов,
        которые имеют хотя бы одно OOB-предсказание.
        '''
        self._get_averaged_oob_predictions()
        mask = self.oob_predictions != None  # Выбираем только те объекты, у которых есть OOB-предсказания
        errors = [(self.target[i] - self.oob_predictions[i]) ** 2 for i in range(len(self.target)) if mask[i]]
        return sum(errors) / len(errors) if errors else None
