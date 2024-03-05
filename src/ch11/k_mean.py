import numpy as np
from sklearn.cluster import KMeans

class KMeanFactory:
    def __init__ (self, k: int, data: np.ndarray):
        self._k = k
        self._data = data
        self._centroids: np.ndarray = None
        self._model: KMeans = None

    def model(self):
        if self._model is None:
            self._model = KMeans(n_clusters=self._k)
            self._model.fit(self._data)                
        
        return self._model
    
    @property
    def k(self):
        return self._k
    
    @property
    def centroids(self):
        if self._centroids is None:
            self._centroids = self.model().cluster_centers_

        return self._centroids
                
def determine_k(data: np.ndarray, max_k: int) -> int:
    start = 1
    clusters = range(start, max_k)
    inertias = [KMeans(n_clusters=i).fit(data).inertia_ for i in clusters]

    # 기울기 절댓값 계산
    gradients = np.diff(inertias)

    # 기울기의 변화율 계산
    gradient_deltas = np.diff(gradients)

    # 변화율의 최대값 인덱스 계산
    max_delta_index = np.argmax(gradient_deltas) + 2 # 0부터 시작하므로 +2

    return max_delta_index + start




