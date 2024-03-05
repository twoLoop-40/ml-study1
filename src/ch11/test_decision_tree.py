from sklearn import datasets, metrics, tree
from sklearn.model_selection import train_test_split
from decision_tree import DecisionTree
import graphviz



# iris data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=50, test_size=0.2)

# Train model
decision_tree = DecisionTree()
decision_tree.data_source = X_train
decision_tree.data_target = y_train

model = decision_tree.train()

y_pred = model.predict(X_test)

score = metrics.accuracy_score(y_test, y_pred)

def test_decision_tree():
    assert score > 0.9

if __name__ == "__main__":
    print('테스트 데이터 정확도:', score)
    try:
        dot_data = tree.export_graphviz(model, out_file=None)
        graph = graphviz.Source(dot_data)
        graph.render('iris')
    except Exception as e:
        print('그래프 생성 오류:', e)
