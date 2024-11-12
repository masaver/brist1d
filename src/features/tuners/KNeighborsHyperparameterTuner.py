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
}

class KNeighborsHyperparameterTuner(BaseHyperparameterTuner):
    __name__ = 'KNeighborsRegressor'

    @staticmethod
    def regressor() -> KNeighborsRegressor:
        return KNeighborsRegressor()

    @staticmethod
    def param_space(search_space: str) -> dict:
        return param_spaces[search_space] if search_space in param_spaces.keys() else param_spaces['default']

    def fit(self, X, y):
        super().fit(X=X, y=y)
