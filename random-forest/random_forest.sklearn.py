# Use SigOpt to tune a Random Forest Classifier in Python
# with the SigOpt + scikit-learn integration
# Learn more about SigOpt + scikit-learn:
# https://github.com/sigopt/sigopt_sklearn
# Learn more about SigOpt's Python Client:
# https://sigopt.com/docs/overview/python

# Run `pip install sigopt_sklearn` to download sigopt_sklearn
from sigopt_sklearn.search import SigOptSearchCV
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
client_token = SIGOPT_API_TOKEN

# Load dataset
# We are using the iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Define domains for the Random Forest parameters
random_forest_parameters = dict(
  max_features=[1, len(iris) - 1],
  n_estimators=[1, 100],
  min_samples_leaf=[1, 10],
)

# define sklearn estimator
random_forest = RandomForestClassifier()

# define SigOptCV search strategy
clf = SigOptSearchCV(
  random_forest,
  random_forest_parameters,
  cv=5,
  client_token=client_token,
  n_iter=60
)

# perform CV search for best parameters and fits estimator
# on all data using best found configuration
clf.fit(X, y)

# clf.predict() now uses best found estimator
# clf.best_score_ contains CV score for best found estimator
# clf.best_params_ contains best found param configuration
