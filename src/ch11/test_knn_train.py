from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from knn_train import KnnTrain
from sklearn import metrics

def test_knn_train():
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 8:2로 데이터를 나눈다.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    knn_train = KnnTrain()
    knn_train.k = 5
    knn_train.data = X_train
    knn_train.target = y_train

    knn_model = knn_train.train()
    assert knn_model is not None

    y_pred = knn_model.predict(X_test)
    scores = metrics.accuracy_score(y_test, y_pred)
    assert scores > 0.95
