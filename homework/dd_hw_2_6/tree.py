import numpy as np
from sklearn.base import BaseEstimator


def entropy(y):  
    """
    Вычисляет энтропию для заданного распределения.  
    Используй log(value + eps) для численной стабильности.

    Параметры
    ----------
    y : np.array типа float с размерностью (n_objects, n_classes)
        One-hot представление меток классов для соответствующего подмножества

    Возвращает
    -------
    float
        Энтропия заданного подмножества
    """
    EPS = 0.0005

    # Вычисляем вероятности для каждого класса
    probabilities = np.sum(y, axis=0) / np.sum(y)
    
    # Вычисляем энтропию
    entropy_value = -np.sum(probabilities * np.log(probabilities + EPS))
    
    return entropy_value
    
def gini(y):
    """
    Вычисляет индекс Джини для заданного распределения.

    Параметры
    ----------
    y : np.array типа float с размерностью (n_objects, n_classes)
        One-hot представление меток классов для соответствующего подмножества.

    Возвращает
    -------
    float
        Индекс Джини для заданного подмножества.
    """
    # Вычисляем вероятности для каждого класса
    probabilities = np.sum(y, axis=0) / np.sum(y)
    
    # Вычисляем индекс Джини
    gini_value = 1 - np.sum(probabilities ** 2)
    
    return gini_value
    
def variance(y):
    """
    Вычисляет дисперсию для заданного подмножества целевых значений.

    Параметры
    ----------
    y : np.array типа float с размерностью (n_objects, 1)
        Вектор целевых значений.

    Возвращает
    -------
    float
        Дисперсия заданного вектора целевых значений.
    """
    # Вычисляем среднее значение целевой переменной
    return np.var(y)

def mad_median(y):
    """
    Вычисляет среднее абсолютное отклонение от медианы  
    для заданного подмножества целевых значений.

    Параметры
    ----------
    y : np.array типа float с размерностью (n_objects, 1)
        Вектор целевых значений.

    Возвращает
    -------
    float
        Среднее абсолютное отклонение от медианы в заданном векторе.
    """
    median = np.median(y)
    return np.mean(np.abs(y - median))

def one_hot_encode(n_classes, y):
    y_one_hot = np.zeros((len(y), n_classes), dtype=float)
    y_indices = y.astype(int).flatten()
    # Гарантируем, что индексы классов в пределах [0, n_classes-1]
    y_indices = np.clip(y_indices, 0, n_classes - 1)
    y_one_hot[np.arange(len(y)), y_indices] = 1.0
    return y_one_hot

def one_hot_decode(y_one_hot):
    return y_one_hot.argmax(axis=1)[:, None]

class Node:
    """
    This class is provided "as is" and it is not mandatory to it use in your code.
    """
    def __init__(self, feature_index, threshold, proba=0):
        self.feature_index = feature_index
        self.value = threshold
        self.proba = proba
        self.left_child = None
        self.right_child = None
               
class DecisionTree(BaseEstimator):
    all_criterions = {
        'gini': (gini, True), # (criterion, classification flag)
        'entropy': (entropy, True),
        'variance': (variance, False),
        'mad_median': (mad_median, False)
    }

    def __init__(self, n_classes=None, max_depth=np.inf, min_samples_split=2, 
                 criterion_name='gini', debug=False):

        assert criterion_name in self.all_criterions.keys(), 'Criterion name must be on of the following: {}'.format(self.all_criterions.keys())
        
        self.n_classes = n_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion_name = criterion_name

        self.depth = 0
        self.root = None # Use the Node class to initialize it later
        self.debug = debug
    
    def make_split(self, feature_index, threshold, X_subset, y_subset):
        """
        Разбивает заданное подмножество данных и целевых значений  
        по указанному признаку и пороговому значению.

        Параметры
        ----------
        feature_index : int
            Индекс признака, по которому выполняется разбиение.

        threshold : float
            Пороговое значение для разбиения.

        X_subset : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, представляющая выбранное подмножество.

        y_subset : np.array типа float с размерностью (n_objects, n_classes) в задаче классификации  
            (n_objects, 1) в задаче регрессии  
            One-hot представление меток классов для соответствующего подмножества.

        Возвращает
        -------
        (X_left, y_left) : кортеж np.array того же типа, что и входные X_subset и y_subset
            Часть заданного подмножества, где выбранный признак x^j < threshold.

        (X_right, y_right) : кортеж np.array того же типа, что и входные X_subset и y_subset
            Часть заданного подмножества, где выбранный признак x^j >= threshold.
        """
        left_mask = X_subset[:, feature_index] < threshold
        right_mask = X_subset[:, feature_index] >= threshold
        
        X_left, y_left = X_subset[left_mask], y_subset[left_mask]
        X_right, y_right = X_subset[right_mask], y_subset[right_mask]
        
        return (X_left, y_left), (X_right, y_right)
    
    def make_split_only_y(self, feature_index, threshold, X_subset, y_subset):
        """
        Разбивает только целевые значения на два подмножества  
        по указанному признаку и пороговому значению.

        Параметры
        ----------
        feature_index : int
            Индекс признака, по которому выполняется разбиение.

        threshold : float
            Пороговое значение для разбиения.

        X_subset : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, представляющая выбранное подмножество.

        y_subset : np.array типа float с размерностью (n_objects, n_classes) в задаче классификации  
            (n_objects, 1) в задаче регрессии  
            One-hot представление меток классов для соответствующего подмножества.

        Возвращает
        -------
        y_left : np.array типа float с размерностью (n_objects_left, n_classes) в задаче классификации  
            (n_objects_left, 1) в задаче регрессии  
            Часть заданного подмножества, где выбранный признак x^j < threshold.

        y_right : np.array типа float с размерностью (n_objects_right, n_classes) в задаче классификации  
            (n_objects_right, 1) в задаче регрессии  
            Часть заданного подмножества, где выбранный признак x^j >= threshold.
        """
        # Маска для объектов, где значение признака меньше порога
        left_mask = X_subset[:, feature_index] < threshold
        
        # Маска для объектов, где значение признака больше или равно порогу
        right_mask = X_subset[:, feature_index] >= threshold
        
        # Разделяем целевые значения на левую и правую части
        y_left = y_subset[left_mask]
        y_right = y_subset[right_mask]
        
        return y_left, y_right

    def choose_best_split(self, X_subset, y_subset):
        """
        Жадным методом выбирает лучший признак и лучший порог в соответствии с выбранным критерием.

        Параметры
        ----------
        X_subset : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, представляющая выбранное подмножество.

        y_subset : np.array типа float с размерностью (n_objects, n_classes) в задаче классификации  
                (n_objects, 1) в задаче регрессии  
            One-hot представление меток классов или целевых значений для соответствующего подмножества.

        Возвращает
        -------
        feature_index : int
            Индекс признака, по которому будет выполняться разбиение.

        threshold : float
            Пороговое значение для разбиения.
        """
        criterion, is_classification = self.all_criterions[self.criterion_name]
        best_feature_index = None
        best_threshold = None
        best_criterion_value = np.inf if is_classification else -np.inf
        
        for feature_index in range(X_subset.shape[1]):
            feature_values = np.unique(X_subset[:, feature_index])
            for threshold in feature_values:
                y_left, y_right = self.make_split_only_y(feature_index, threshold, X_subset, y_subset)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                
                # Для регрессии используем исходные значения y
                left_criterion = criterion(y_left)
                right_criterion = criterion(y_right)
                total_objects = len(y_left) + len(y_right)
                weighted_criterion = (len(y_left) * left_criterion + len(y_right) * right_criterion) / total_objects
                
                if (is_classification and weighted_criterion < best_criterion_value) or \
                (not is_classification and weighted_criterion > best_criterion_value):
                    best_criterion_value = weighted_criterion
                    best_feature_index = feature_index
                    best_threshold = threshold
        
        return best_feature_index, best_threshold
    
    def make_tree(self, X_subset, y_subset, depth=0):
        """
        Рекурсивно строит дерево.

        Параметры
        ----------
        X_subset : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, представляющая выбранное подмножество.

        y_subset : np.array типа float с размерностью (n_objects, n_classes) в задаче классификации  
                (n_objects, 1) в задаче регрессии  
            One-hot представление меток классов или целевых значений для соответствующего подмножества.

        Возвращает
        -------
        root_node : Экземпляр класса Node
            Узел корня построенного дерева.
        """
        criterion, is_classification = self.all_criterions[self.criterion_name]
        
        # Условие остановки: максимальная глубина или мало объектов
        if depth >= self.max_depth or len(X_subset) < self.min_samples_split:
            if is_classification:
                # Для классификации: вычисляем вероятности классов
                class_counts = np.sum(y_subset, axis=0)
                total = np.sum(class_counts)
                if total == 0:
                    proba = np.ones(self.n_classes) / self.n_classes
                else:
                    proba = class_counts / total
                return Node(None, None, proba=proba)
            else:
                # Для регрессии: возвращаем среднее или медиану
                if self.criterion_name == "variance":
                    return Node(None, np.mean(y_subset))
                elif self.criterion_name == "mad_median":
                    return Node(None, np.median(y_subset))
        
        # Выбираем лучшее разделение
        feature_index, threshold = self.choose_best_split(X_subset, y_subset)
        
        # Если не удалось найти разделение, создаем листовой узел
        if feature_index is None:
            if is_classification:
                class_counts = np.sum(y_subset, axis=0)
                total = np.sum(class_counts)
                proba = class_counts / total if total != 0 else np.ones(self.n_classes) / self.n_classes
                return Node(None, None, proba=proba)
            else:
                if self.criterion_name == "variance":
                    return Node(None, np.mean(y_subset))
                elif self.criterion_name == "mad_median":
                    return Node(None, np.median(y_subset))
        
        # Создаем новый узел
        new_node = Node(feature_index, threshold)
        (X_left, y_left), (X_right, y_right) = self.make_split(feature_index, threshold, X_subset, y_subset)
        new_node.left_child = self.make_tree(X_left, y_left, depth + 1)
        new_node.right_child = self.make_tree(X_right, y_right, depth + 1)
        return new_node
        
    def fit(self, X, y):
        """
        Обучает модель с нуля, используя предоставленные данные.

        Параметры
        ----------
        X : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, представляющая данные для обучения.

        y : np.array типа int с размерностью (n_objects, 1) в задаче классификации  
                или типа float с размерностью (n_objects, 1) в задаче регрессии  
            Вектор столбца с метками классов в классификации или целевыми значениями в регрессии.
        """

        assert len(y.shape) == 2 and len(y) == len(X), 'Wrong y shape'
        self.criterion, self.classification = self.all_criterions[self.criterion_name]
        if self.classification:
            if self.n_classes is None:
                self.n_classes = len(np.unique(y))
            y = one_hot_encode(self.n_classes, y)

        self.root = self.make_tree(X, y)
    
    def predict(self, X):
        """
        Предсказывает целевое значение или метку класса, используя модель, обученную с нуля, на предоставленных данных.

        Параметры
        ----------
        X : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, для которой должны быть сделаны предсказания.

        Возвращает
        -------
        y_predicted : np.array типа int с размерностью (n_objects, 1) в задаче классификации  
                    (n_objects, 1) в задаче регрессии  
            Вектор столбца с предсказанными метками классов в классификации или целевыми значениями в регрессии.
        """
        y_predicted = []
        
        for sample in X:
            node = self.root
            while node.left_child or node.right_child:
                if sample[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
            # Если это классификация, возвращаем метку класса с наибольшей вероятностью
            if self.classification:
                predicted_class = np.argmax(node.proba)
                y_predicted.append(predicted_class)
            else:
                # Если это регрессия, возвращаем предсказанное значение
                y_predicted.append(node.value)
        
        return np.array(y_predicted).reshape(-1, 1)
        
    def predict_proba(self, X):
        """
        Только для классификации.
        Предсказывает вероятности классов, используя предоставленные данные.

        Параметры
        ----------
        X : np.array типа float с размерностью (n_objects, n_features)
            Матрица признаков, для которой должны быть сделаны предсказания вероятностей.

        Возвращает
        -------
        y_predicted_probs : np.array типа float с размерностью (n_objects, n_classes)
            Вероятности каждого класса для предоставленных объектов.
        """

        assert self.classification, 'Available only for classification problem'

        y_predicted_probs = []
        
        for sample in X:
            node = self.root
            while node.left_child or node.right_child:
                if sample[node.feature_index] < node.value:
                    node = node.left_child
                else:
                    node = node.right_child
            # Вероятности классов из листового узла
            y_predicted_probs.append(node.proba)
        
        return np.array(y_predicted_probs)
