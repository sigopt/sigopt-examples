import orchestrate.io
from sklearn import datasets
from sklearn.linear_model  import SGDClassifier
from sklearn.model_selection import cross_val_score
import numpy


def load_data():
  iris = datasets.load_iris()
  return (iris.data, iris.target)


def evaluate_model(X, y):
  classifier = SGDClassifier(
    loss=orchestrate.io.assignment('loss', default='log'),
    penalty=orchestrate.io.assignment('penalty', default='elasticnet'),
    alpha=10**orchestrate.io.assignment('log_alpha', -4),
    l1_ratio=orchestrate.io.assignment('l1_ratio', 0.15),
    max_iter=orchestrate.io.assignment('max_iter', default=1000),
    tol=orchestrate.io.assignment('tol', default=0.001),
  )
  cv_accuracies = cross_val_score(classifier, X, y, cv=5)
  return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

if __name__ == "__main__":
  (X, y) = load_data()
  (mean, std) = evaluate_model(X=X, y=y)
  print('Accuracy: {} +/- {}'.format(mean, std))
  orchestrate.io.log_metric('accuracy', mean, std)
