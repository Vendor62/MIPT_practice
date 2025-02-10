import numpy as np
"""
Credits: исходный код принадлежит курсовому заданию Stanford CS231n. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""

class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Обучение классификатора.  
        Для метода k ближайших соседей (k-nearest neighbors) это просто запоминание обучающих данных.  

        Входные данные:  
        - X: Numpy-массив формы (num_train, D), содержащий обучающие данные,  
        состоящие из num_train образцов, каждый размерностью D.  
        - y: Numpy-массив формы (N,), содержащий метки классов,  
        где y[i] — метка для X[i].  

        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Предсказание меток для тестовых данных с использованием этого классификатора.  

        Входные данные:  
        - X: Numpy-массив формы (num_test, D), содержащий тестовые данные,  
        состоящие из num_test образцов, каждый размерностью D.  
        - k: Количество ближайших соседей, которые участвуют в голосовании  
        для предсказания метки.  
        - num_loops: Определяет, какая реализация будет использована  
        для вычисления расстояний между обучающими и тестовыми точками.  

        Выходные данные:  
        - y: Numpy-массив формы (num_test,), содержащий предсказанные метки  
        для тестовых данных, где y[i] — предсказанная метка для тестовой точки X[i].  

        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой из X и каждой  
        обучающей точкой из self.X_train с использованием вложенного цикла  
        по обучающим и тестовым данным.  

        Входные данные:  
        - X: Numpy-массив формы (num_test, D), содержащий тестовые данные.  

        Выходные данные:  
        - dists: Numpy-массив формы (num_test, num_train), где dists[i, j]  
        — это евклидово расстояние между i-й тестовой точкой и j-й  
        обучающей точкой.  
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Вычислить L2-расстояние между i-й тестовой точкой и j-й           #
                # обучающей точкой и сохранить результат в dists[i, j].             #
                # Нельзя использовать цикл по измерениям и np.linalg.norm().        #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                diff = X[i, :] - self.X_train[j, :]  # разница между тестовой и обучающей точкой
                dists[i, j] = np.sqrt(np.sum(diff ** 2)) # вычисляем сумму квадратов и извлекаем квадратный корень
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой из X и каждой  
        обучающей точкой из self.X_train, используя один цикл по тестовым данным.  

        Входные / выходные данные: такие же, как в compute_distances_two_loops. 
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            ########################################################################
            # TODO:                                                                #
            # Вычислить L2-расстояние между i-й тестовой точкой и всеми обучающими #
            # точками и сохранить результат в dists[i, :].                         #
            # Не использовать np.linalg.norm().                                    #
            ########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            diff = self.X_train - X[i, :]  # разница между i-й тестовой и всеми обучающими точками
            dists[i, :] = np.sqrt(np.sum(diff ** 2, axis=1))  # сумма квадратов разностей по всем признакам, квадратный корень
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Вычисляет расстояние между каждой тестовой точкой из X и каждой  
        обучающей точкой из self.X_train без явных циклов.  

        Входные / выходные данные: такие же, как в compute_distances_two_loops.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Вычислить L2-расстояние между всеми тестовыми и всеми обучающими      #
        # точками без использования явных циклов и сохранить результат в dists. #
        # Реализовать эту функцию, используя только базовые операции с          #
        # массивами; в частности, нельзя использовать функции из scipy и        #
        # np.linalg.norm().                                                     #      
        # ПОДСКАЗКА: попробуй выразить L2-расстояние через умножение матриц     #
        # и два broadcast-сложения.                                             #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # Шаг 1: Вычислим квадрат суммы разностей между тестовыми и обучающими точками.
        # (X ** 2) - это квадрат всех значений в тестовых данных.
        # (self.X_train ** 2) - это квадрат всех значений в обучающих данных.
        # (X @ X.T) - это умножение тестовых данных на их транспонированную матрицу.
    
        # (X ** 2).sum(axis=1) суммирует квадратные значения для каждой тестовой точки по всем признакам.
        # (self.X_train ** 2).sum(axis=1) суммирует квадратные значения для каждой обучающей точки.
    
        # Шаг 2: Для каждого тестового примера вычислим евклидово расстояние до всех обучающих точек.
        dists = np.sqrt(np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(self.X_train ** 2, axis=1) - 2 * X.dot(self.X_train.T))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def predict_labels(self, dists, k=1):
        """
        Дана матрица расстояний между тестовыми и обучающими точками,  
        необходимо предсказать метку для каждой тестовой точки.  

        Входные данные:  
        - dists: Numpy-массив формы (num_test, num_train), где dists[i, j]  
        представляет расстояние между i-й тестовой точкой и j-й обучающей точкой.  

        Выходные данные:  
        - y: Numpy-массив формы (num_test,), содержащий предсказанные метки  
        для тестовых данных, где y[i] — предсказанная метка для тестовой точки X[i]. 
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # Список длины k, хранящий метки k ближайших соседей для  
            # i-й тестовой точки.  
            #########################################################################
            # TODO:                                                                 #
            # Используйте матрицу расстояний, чтобы найти k ближайших соседей       #
            # для i-й тестовой точки, и используйте self.y_train, чтобы найти метки #
            # этих соседей. Сохраните эти метки в closest_y.                        #
            # Подсказка: посмотрите функцию numpy.argsort.                          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            # Находим индексы k ближайших соседей для i-й тестовой точки
            closest_y_indices = np.argsort(dists[i])[:k] 
            # Получаем метки этих k ближайших соседей
            closest_y = self.y_train[closest_y_indices]
            # Находим наиболее частую метку среди k ближайших соседей
            counts = np.bincount(closest_y)
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Теперь, когда вы нашли метки k ближайших соседей, нужно найти         #
            # наиболее частую метку в списке closest_y с метками.                   #
            # Сохраните эту метку в y_pred[i]. В случае ничьей выберите меньшую     #
            # метку.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
             # Получаем метку с наибольшим количеством
            y_pred[i] = np.min(np.where(counts == np.max(counts)))
            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred