from xgboost import XGBClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sigopt

iris = load_iris()
X = iris.data
Y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

def create_model():
    model = XGBClassifier(
      learning_rate=10**sigopt.get_parameter('log_learning_rate', default=-3),
      max_depth=sigopt.get_parameter('max_depth', default=3),
      num_rounds=sigopt.get_parameter('num_rounds', default=10),
      min_child_weight=sigopt.get_parameter('min_child_weight', default=5),
      objective='binary:logistic',
    )
    model.fit(X_train, y_train)

    return model

def evaluate_model():
    model = create_model()
    pred = model.predict(X_test)
    return accuracy_score(pred, y_test)

if __name__ == '__main__':
    sigopt.log_metric('accuracy', evaluate_model())
