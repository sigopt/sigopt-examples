import sigopt
from data_and_model_setup import LoadTransformData, log_inference_metrics
import time
import platform
from xgboost.sklearn import XGBClassifier


def train_xgboost_model(dataset, random_state=1):
    print("loading and transforming data")
    load_transform_data = LoadTransformData()
    trainX, testX, trainY, testY = load_transform_data.load_split_dataset(dataset)

    # model architecture
    sigopt.log_model("XGBClassifier")  # model_keras.__class__
    sigopt.log_dataset('Unscaled')
    sigopt.log_metadata('Training Records', len(trainX))
    sigopt.log_metadata('Testing Reccords', len(testX))
    sigopt.log_metadata("Platform", platform.uname())

    parameters = {
        'objective': 'binary:logistic',
        'learning_rate': sigopt.get_parameter('learning_rate', default=0.3),
        'n_estimators': sigopt.get_parameter('n_estimators', default=20),
        'max_depth': sigopt.get_parameter('max_depth', default=5),
        'gamma': sigopt.get_parameter('gamma', default=0),
        'min_child_weight': sigopt.get_parameter('min_child_weight', default=1),
        'random_state': random_state,
        'importance_type': 'gain',
        'missing': None,
        'verbosity': 2}

    model = XGBClassifier(**parameters)

    modelfit = model.fit(trainX, trainY)

    # Collect model metrics
    start = time.perf_counter()
    prediction = modelfit.predict(testX)
    sigopt.log_metric("Inference Time", time.perf_counter() - start)
    probability = modelfit.predict_proba(testX)[:, 1]
    log_inference_metrics(prediction, probability, testY, testX)


if __name__ == "__main__":
    dataset_file = 'https://www.dropbox.com/s/437qdt4yjj64sxd/Fraud_Detection_SigOpt_dataset.csv?dl=1'
    train_xgboost_model(dataset_file)
