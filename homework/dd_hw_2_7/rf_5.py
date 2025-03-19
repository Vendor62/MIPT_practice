import numpy as np
import pandas as pd

np.random.seed(42)  # Фиксация случайности

def get_bootstrap(data, labels, n=100, max_samples=1.0):
    if isinstance(data, pd.DataFrame):  
        data = data.copy().values  # Перевод таблицы в numpy array

    if isinstance(labels, pd.DataFrame) or isinstance(labels, pd.Series):
        labels = labels.copy().values  # Перевод целевого признака в numpy array

    n_samples = int(max_samples * len(data))  # Подсчет кол-ва объектов в подвыборках
    bootstrap = []  # Список для будущих подвыборок

    for i in range(n):
        sample_index = np.random.choice(len(data), n_samples, replace=True)  # Случайные индексы
        b_data = data[sample_index]  # Объекты по выбранным индексам
        b_labels = labels[sample_index]  # Целевые значения по выбранным индексам

        bootstrap.append((b_data, b_labels))  # Добавление в список подвыборок

    return bootstrap

def get_subsample(features, max_features=1.0):
    if not isinstance(features, np.ndarray):
        features = np.array(features.copy())  # Перевод признаков в numpy array

    len_features = int(max_features * len(features))  # Подсчет кол-ва признаков в подвыборке
    sample_indexes = list(range(len(features)))  # Получение списка индексов признаков

    subsample = np.random.choice(
        sample_indexes,  # Индексы признаков
        size=len_features,  # Количество признаков для выборки
        replace=False  # Без повторений
    )

    return features[subsample] 




'''
Пример работы функции на данных из двух признаков на первом запуске:

sample_X = np.array([
    [0, 0],
    [1, 1],
    [2, 2]
])

sample_y = np.array([0, 1, 2])

get_bootstrap(sample_X, sample_y, n=3, max_samples=1.0)

Output:

[(array([[2, 2],
         [0, 0],
         [2, 2]]),
  array([2, 0, 2])),
 (array([[2, 2],
         [0, 0],
         [0, 0]]),
  array([2, 0, 0])),
 (array([[2, 2],
         [1, 1],
         [2, 2]]),
  array([2, 1, 2]))]
'''
