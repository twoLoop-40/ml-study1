import numpy as np
from sklearn.tree import DecisionTreeClassifier

class DecisionTree:
    def __init__(self):
        self._model = None
        self._data_source: np.ndarray = None
        self._data_target: np.ndarray = None

    @property
    def data_source(self) -> np.ndarray | None:
        return self._data_source
    
    @data_source.setter
    def data_source(self, data: np.ndarray):
        self._data_source = np.array(data)

    @property
    def data_target(self):
        return self._data_target
    
    @data_target.setter
    def data_target(self, data: np.ndarray):
        self._data_target = np.array(data)

    def train(self) -> DecisionTreeClassifier | None:
        if self._data_source is None:
            raise ValueError('데이터 입력 전 입니다. 데이터를 입력해주세요.')
        
        if self._data_target is None:
            raise ValueError('타겟값 입력 전 입니다. 타겟값을 입력해주세요.')

        if self._model is None:
            self._model = DecisionTreeClassifier()
        
        self._model.fit(self._data_source, self._data_target)
        return self._model