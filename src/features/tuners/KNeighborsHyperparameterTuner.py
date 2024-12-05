from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Integer, Categorical

from src.features.tuners.BaseHyperparameterTuner import BaseHyperparameterTuner

param_spaces = {
    'default': {
        'n_neighbors': Integer(1, 30),  # Number of neighbors to consider, covering both small and large values
        'weights': Categorical(['uniform', 'distance']),  # Weighting of neighbors, with 'distance' giving closer neighbors more weight
        'p': Categorical([1, 2]),  # Power parameter for Minkowski distance, with 1 being Manhattan and 2 being Euclidean
        'leaf_size': Integer(10, 50),  # Leaf size for the tree-based neighbor search, affects efficiency
    },
    'custom': {
        'n_neighbors': Integer(3, 15),  # Extended to include smaller k values
        'weights': Categorical(['uniform', 'distance']),  # Added 'distance' weighting
        'p': Categorical([1, 2]),  # Added Manhattan distance
        'leaf_size': Integer(10, 50),  # Broader range for flexibility
        'metric': Categorical(['minkowski', 'euclidean', 'chebyshev']),  # Added diverse distance metrics
    }
}


class KNeighborsHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'KNeighborsRegressor'

    @staticmethod
    def regressor() -> KNeighborsRegressor:
        return KNeighborsRegressor()

    @staticmethod
    def param_space(search_space: str | None) -> dict | None:
        if search_space is None:
            return None
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X_train, y_train, X_test=None, y_test=None):
        super().fit(X_train, y_train, X_test, y_test)
