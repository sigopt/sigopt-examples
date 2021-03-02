# SGDClassifier example using SigOpt IO

import sigopt
from sklearn import datasets
from sklearn.linear_model  import SGDClassifier
from sklearn.model_selection import cross_val_score
import numpy


def load_data():
  iris = datasets.load_iris()
  return (iris.data, iris.target)


def evaluate_model(X, y):
  classifier = SGDClassifier(
    loss=sigopt.get_parameter('loss', default='log'),
    penalty=sigopt.get_parameter('penalty', default='elasticnet'),
    alpha=10**sigopt.get_parameter('log_alpha', -4),
    l1_ratio=sigopt.get_parameter('l1_ratio', 0.15),
    max_iter=sigopt.get_parameter('max_iter', default=1000),
    tol=sigopt.get_parameter('tol', default=0.001),
  )
  cv_accuracies = cross_val_score(classifier, X, y, cv=5)
  return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

if __name__ == "__main__":
  (X, y) = load_data()
  (mean, std) = evaluate_model(X=X, y=y)
  print('Accuracy: {} +/- {}'.format(mean, std))
  sigopt.log_metric('accuracy', mean, std)
