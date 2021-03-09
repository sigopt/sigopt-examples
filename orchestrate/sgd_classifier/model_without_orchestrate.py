# SGDClassifier example without SigOpt IO


from sklearn import datasets
from sklearn.linear_model  import SGDClassifier
from sklearn.model_selection import cross_val_score
import numpy


def load_data():
  iris = datasets.load_iris()
  return (iris.data, iris.target)


def evaluate_model(X, y):
  classifier = SGDClassifier(
    loss='log',
    penalty='elasticnet',
    alpha=10**(-4),
    l1_ratio=0.15,
    max_iter=1000,
    tol=0.001,
  )
  cv_accuracies = cross_val_score(classifier, X, y, cv=5)
  return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

if __name__ == "__main__":
  (X, y) = load_data()
  (mean, std) = evaluate_model(X=X, y=y)
  print('Accuracy: {} +/- {}'.format(mean, std))
