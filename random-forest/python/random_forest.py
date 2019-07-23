# Use SigOpt to tune a Random Forest Classifier in Python
# Learn more about SigOpt's Python Client:
# https://sigopt.com/docs/overview/python

# Run `pip install sigopt` to download the python API client
# Run `pip install sklearn` to install scikit-learn, a machine learning
# library in Python (http://scikit-learn.org)
from sigopt import Connection
from sklearn import datasets
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import numpy

# Learn more about authenticating the SigOpt API:
# https://sigopt.com/docs/overview/authentication
conn = Connection(client_token=SIGOPT_API_TOKEN)

# Load dataset
# We are using the iris dataset as an example
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a SigOpt experiment for the Random Forest parameters
experiment = conn.experiments().create(
  name="Random Forest (Python)",
  project="sigopt-examples",
  metrics=[dict(name='accuracy', objective='maximize')],
  parameters=[
    dict(name="max_features", type="int", bounds=dict(min=1, max=len(iris)-2)),
    dict(name="n_estimators", type="int", bounds=dict(min=1, max=100)),
    dict(name="min_samples_leaf", type="int", bounds=dict(min=1, max=10))
  ]
)
print("Created experiment: https://sigopt.com/experiment/" + experiment.id)

# Our object metric is the mean of cross validated accuracies
# We use cross validation to prevent overfitting
def evaluate_model(assignments, X, y):
  # evaluate cross folds for accuracy
  cv = ShuffleSplit(
    n_splits=5,
    test_size=0.3,
  )
  classifier = RandomForestClassifier(
    n_estimators=assignments['n_estimators'],
    max_features=assignments['max_features'],
    min_samples_leaf=assignments['min_samples_leaf']
  )
  cv_accuracies = cross_val_score(classifier, X, y, cv=cv)
  return (numpy.mean(cv_accuracies), numpy.std(cv_accuracies))

# Run the Optimization Loop between 10x - 20x the number of parameters
for _ in range(60):
    # Receive a Suggestion from SigOpt
    suggestion = conn.experiments(experiment.id).suggestions().create()

    # Evaluate the model locally
    (value, std) = evaluate_model(suggestion.assignments, X, y)

    # Report an Observation (with standard deviation) back to SigOpt
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=value,
        value_stddev=std,
    )

# Re-fetch the best observed value and assignments
best_assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments

# To wrap up the Experiment, fit the RandomForest on the best assignments
# and train on all available data
rf = RandomForestClassifier(
    n_estimators=best_assignments['n_estimators'],
    max_features=best_assignments['max_features'],
    min_samples_leaf=best_assignments['min_samples_leaf']
)
rf.fit(X, y)
