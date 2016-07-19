# SigOpt Random Forest Example

This example tunes a random forest using SigOpt on the open IRIS dataset. Scroll down to find your language!

## Python
Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in line 16 of `random_forest.py`, then run the following code in a terminal to install dependencies and execute the script:

```
pip install sigopt
pip install sklearn
cd python
python random_forest.py
```

Learn more about our [Python API Client](https://sigopt.com/docs/overview/python).

## R
Simply add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in line 16 of `random_forest_SigOpt.R` and then execute the R script in R Studio, or in the terminal:

```
RScript random_forest_SigOpt.R
```

Learn more about our [R API Client](https://sigopt.com/docs/overview/r).

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

## Scikit-learn Integration
Add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) in line 15 of `random_forest.sklearn.py`, then run the following code in a terminal to install dependencies and execute the script:

```
pip install sigopt_sklearn
cd python
python random_forest.sklearn.py
```

Learn more about our [scikit-learn integration](https://github.com/sigopt/sigopt_sklearn).

## Jupyter iPython Notebook
We have a version of our Python Random Forest Example in a convenient [iPython notebook](https://ipython.org/) form.
To run, start up the notebook and add your SigOpt [API token](https://sigopt.com/docs/overview/authentication) into the first cell. Run cells with `shift + enter`.

To run the notebook:

```
pip install jupyter
cd python
jupyter notebook
```

Learn more about our [Python API Client](https://sigopt.com/docs/overview/python) that we use in the notebook.

## Questions?
Any questions? Drop us a line at [support@sigopt.com](mailto:support@sigopt.com).
