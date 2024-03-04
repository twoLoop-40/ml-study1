from sklearn import metrics
from sklearn.model_selection import train_test_split
from digit_data import Digit
from knn_train import KnnTrain

# digit data load
digit = Digit()

# 8:2로 데이터를 나눈다.
X_train, X_test, y_train, y_test = train_test_split(digit.digits, digit.target, test_size=0.2)

# knn model train
trainer = KnnTrain()
trainer.k = 6
trainer.data = X_train
trainer.target = y_train

model = trainer.train()

# test
y_pred = model.predict(X_test)

# 정확도 확인
scores = metrics.accuracy_score(y_test, y_pred)
assert scores < 0.95