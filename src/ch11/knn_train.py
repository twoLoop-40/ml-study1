import numpy as np
import typing as T
from sklearn.neighbors import KNeighborsClassifier 

class KnnTrain:
    def __init__ (self):
        self._k = None
        self._data = np.array([])
        self._target = np.array([])

    @property    
    def k(self):
        if self._k is None:
            raise ValueError('k is not set')
        return self._k
    
    @k.setter
    def k(self, k):
        self._k = k

    @property
    def data(self) -> np.ndarray:
        if len(self._data) == 0:
            raise ValueError('데이터 입력 전 입니다. 데이터를 입력해주세요.')
        else:
            return self._data

    @data.setter
    def data(self, newData: np.ndarray):
        self._data = np.array(newData)

    @property
    def target(self):
        if len(self._target) == 0:
            raise ValueError('타겟값 입력 전 입니다. 타겟값을 입력해주세요.')
        else:
            return self._target
        
    @target.setter
    def target(self, newTarget: T.List[float]):
        self._target = np.array(newTarget)

    def train(self):
        try:
            knn = KNeighborsClassifier(n_neighbors=self.k)
            knn.fit(self.data, self.target)
            return knn

        except ValueError as e:
            print(e)
            return


    