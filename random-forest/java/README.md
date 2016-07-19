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
