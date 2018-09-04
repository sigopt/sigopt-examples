import orchestrate.io
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy


def load_data():
  iris = datasets.load_iris()
  return (iris.data, iris.target)


def evaluate_model(X, y):
  classifier = RandomForestClassifier(
    n_estimators=orchestrate.io.assignment('n_estimators', default=3),
    max_features=orchestrate.io.assignment('max_features', default=3),
    min_samples_leaf=orchestrate.io.assignment('min_samples_leaf', default=80)
  )
  cv_accuracies = cross_val_score(classifier, X, y, cv=5)
  return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

if __name__ == "__main__":
  (X, y) = load_data()
  (mean, std) = evaluate_model(X=X, y=y)
  orchestrate.io.log_metric('accuracy', mean, std)
