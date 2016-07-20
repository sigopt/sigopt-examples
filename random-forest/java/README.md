# SigOpt Random Forest Java Example

This example tunes a random forest using SigOpt + Java on the open IRIS dataset.

## Java
You will need [maven](https://maven.apache.org/) to compile and run this example. On OS X, you can install maven with Homebrew by running

```bash
brew install maven
```

Next, compile

```bash
mvn compile
```
Insert your SigOpt [API token](https://sigopt.com/docs/overview/authentication) into the command below to run the example

```bash
mvn exec:java -Dexec.mainClass="com.example.RandomForestApp" -Dexec.cleanupDaemonThreads="false" -Dexec.args="--api_token $SIGOPT_API_TOKEN"
```

To build this example we used [Weka](http://www.cs.waikato.ac.nz/ml/weka/), a collection of machine learning algorithms in java.

Learn more about our [Java API Client](https://sigopt.com/docs/overview/java).

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available for a [30 day free trial](https://sigopt.com/signup), and is avaialable [free forever for academic users](https://sigopt.com/edu).
