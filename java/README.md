[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

Use SigOpt's [Java API Client](https://sigopt.com/docs/overview/java) to tune the [Franke function](http://www.sfu.ca/~ssurjano/franke2d.html)

# Installation
You will need [maven](https://maven.apache.org/) to compile and run this example.
On OS X, you can install maven with Homebrew by running

```bash
brew install maven
```

# Compile

```bash
mvn compile
```

# Run
Insert your SigOpt API Token into the example below

```bash
mvn exec:java -Dexec.mainClass="com.example.App" -Dexec.cleanupDaemonThreads="false" -Dexec.args="--client_token SIGOPT_API_TOKEN"
```

## Questions?
Visit the [SigOpt Community page](https://community.sigopt.com) and leave your questions.

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible.
## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API and client libraries (Python, R, Java) integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

[Visit our website](https://sigopt.com) to learn more!
