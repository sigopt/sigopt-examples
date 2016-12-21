from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import numpy
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

PARAMETERS = [
  dict(name="max_features", type="int", bounds=dict(min=1, max=len(iris)-1)),
  dict(name="n_estimators", type="int", bounds=dict(min=1, max=100)),
  dict(name="min_samples_leaf", type="int", bounds=dict(min=1, max=10))
]

def evaluate_model(assignments):
  cv = cross_validation.ShuffleSplit(
    X.shape[0],
    n_iter=5,
    test_size=0.3,
  )
  classifier = RandomForestClassifier(
    n_estimators=assignments['n_estimators'],
    max_features=assignments['max_features'],
    min_samples_leaf=assignments['min_samples_leaf']
  )
  cv_accuracies = cross_validation.cross_val_score(classifier, X, y, cv=cv)
  return numpy.mean(cv_accuracies)
