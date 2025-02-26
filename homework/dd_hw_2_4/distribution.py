import numpy as np

class LaplaceDistribution:    
    @staticmethod
    def mean_abs_deviation_from_median(x: np.ndarray):
        '''
        Args:
        - x: A numpy array of shape (n_objects, n_features) containing the data
          consisting of num_train samples each of dimension D.
        '''
        ####
        # Do not change the class outside of this block
        # Your code here
        median = np.median(x, axis=0)  # Медиана по каждому признаку
        mad = np.mean(np.abs(x - median), axis=0)  # Среднее абсолютное отклонение от медианы
        return mad
        ####

    def __init__(self, features):
        '''
        Args:
            features: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        self.loc = np.median(features, axis=0)  # Медиана по каждому признаку
        self.scale = self.mean_abs_deviation_from_median(features)  # Среднее абсолютное отклонение
        ####


    def logpdf(self, values):
        '''
        Returns logarithm of probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        ####
        # Do not change the class outside of this block
        # Логарифм плотности вероятности для каждого значения
        return -np.log(2 * self.scale) - np.abs(values - self.loc) / self.scale
        ####
        
    
    def pdf(self, values):
        '''
        Returns probability density at every input value.
        Args:
            values: A numpy array of shape (n_objects, n_features). Every column represents all available values for the selected feature.
        '''
        return np.exp(self.logpdf(values))  # Вычисление плотности вероятности

