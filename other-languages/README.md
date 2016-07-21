# SigOpt with Other Languages
Our `other_languages` example is one way to use SigOpt when your metric evaluation function is in a language other than python. All you need to do is create an executable file that accepts parameters as command line arguments, and then create an experiment with the same parameter names as the executable. The executable file should accept the suggested parameters at the command line, evaluate the metric, and print out a float.

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile).
3. `export CLIENT_TOKEN=<your client_token>`
4. Install the SigOpt python client `pip install sigopt`

## Example Usage
```
python other_languages.py --command='<command to run your script>' --experiment_id=EXPERIMENT_ID --client_token=$CLIENT_TOKEN
```
The above command will run the following sub process to evaluate your metric, automatially requesting the suggsetion beforehand and reporting the observation afterwards:
```
<comamnd to run your script> --x=SUGGESTION_FOR_X
```

Feel free to use, or modify for your own needs!

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).

## API Reference
To implement SigOpt for your use case, feel free to use or extend the code in this repository. Our [core API](https://sigopt.com/docs) can bolt on top of any complex model or process and guide it to its optimal configuration in as few iterations as possible. 

## About SigOpt

With SigOpt, data scientists and machine learning engineers can build better models with less trial and error.

Machine learning models depend on hyperparameters that trade off bias/variance and other key outcomes. SigOpt provides Bayesian hyperparameter optimization using an ensemble of the latest research.

SigOpt can tune any machine learning model, including popular techniques like gradient boosting, deep neural networks, and support vector machines. SigOpt’s REST API, Python, and R libraries integrate into any existing ML workflow.

SigOpt augments your existing model training pipeline, suggesting parameter configurations to maximize any online or offline objective, such as AUC ROC, model accuracy, or revenue. You only send SigOpt your metadata, not the underlying training data or model.

SigOpt is available for a [30 day free trial](https://sigopt.com/signup), and is available [free forever for academic users](https://sigopt.com/edu).
