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
