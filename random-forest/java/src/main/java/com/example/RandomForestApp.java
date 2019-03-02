// Use SigOpt to tune a Random Forest Classifier in Java
// Learn more about SigOpt's Java Client:
// https://sigopt.com/docs/overview/java

package com.example;

import com.example.Result;

import java.util.Arrays;
import java.util.Random;

import com.sigopt.Sigopt;
import com.sigopt.exception.APIException;
import com.sigopt.model.Assignments;
import com.sigopt.model.Bounds;
import com.sigopt.model.Experiment;
import com.sigopt.model.Observation;
import com.sigopt.model.Parameter;
import com.sigopt.model.Suggestion;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.Prediction;
import weka.classifiers.trees.RandomForest;

public class RandomForestApp
{
    public static void main(String[] args) throws java.lang.Exception, APIException
    {
        // Learn more about authenticating the SigOpt API:
        // https://sigopt.com/docs/overview/authentication
        String clientId = null;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--api_token") && i < args.length - 1) {
                Sigopt.clientToken = args[i+1];
            }
        }

        // Load dataset
        // We are using the iris dataset as an example
        DataSource irisSource = new DataSource("iris.arff");
        Instances irisData = irisSource.getDataSet();
        if (irisData.classIndex() == -1) {
           irisData.setClassIndex(irisData.numAttributes() - 1);
        }

        // Create a SigOpt experiment for the Random Forest parameters
        Experiment experiment = Experiment.create(
            new Experiment.Builder()
                .name("Random Forest (Java)")
                .parameters(Arrays.asList(
                    new Parameter.Builder()
                        .name("maxDepth")
                        .type("int")
                        .bounds(new Bounds(1.0, 50.0))
                        .build(),
                    new Parameter.Builder()
                        .name("bagSizePercent")
                        .type("int")
                        .bounds(new Bounds(1.0, 100.0))
                        .build(),
                    new Parameter.Builder()
                        .name("numIterations")
                        .type("int")
                        .bounds(new Bounds(1.0, 100.0))
                        .build()
                ))
                .build())
            .call();
        System.out.println("Created experiment: https://sigopt.com/experiment/" + experiment.getId());

        // Run the Optimization Loop between 10x - 20x the number of parameters
        for (int i = 0; i < 60; i++) {
            // Receive a Suggestion from SigOpt
            Suggestion suggestion = experiment.suggestions().create().call();

            // Evaluate the model locally
            Result result = RandomForestApp.evaluateModel(suggestion.getAssignments(), irisData);

            // Report an Observation (with standard deviation) back to SigOpt
            Observation observation = experiment.observations().create()
                .data(new Observation.Builder()
                    .suggestion(suggestion.getId())
                    .value(result.value)
                    .valueStddev(result.stddev)
                    .build())
                .call();
        }

        // Re-fetch the experiment to get the best observed value and assignments
        experiment = Experiment.fetch(experiment.getId()).call();
        Assignments bestAssignments = experiment.getProgress().getBestObservation().getAssignments();

        // To wrap up the Experiment, fit the RandomForest on the best assignments and train on all available data
        RandomForest bestRandomForest = RandomForestApp.randomForestFromAssignments(bestAssignments);
        bestRandomForest.buildClassifier(irisData);
    }

    // Our object metric is cross validated accuracy
    // We use cross validation to prevent overfitting
    public static Result evaluateModel(Assignments assignments, Instances data) throws java.lang.Exception
    {
        RandomForest rf = RandomForestApp.randomForestFromAssignments(assignments);

        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(rf, data, 5, new Random(1));

        double accuracy = eval.pctCorrect() / 100.0;
        return new Result(accuracy, null);
    }

    // Create a new RandomForest and set the hyperparameters from the assignments
    public static RandomForest randomForestFromAssignments(Assignments assignments) {
        RandomForest rf = new RandomForest();
        rf.setMaxDepth(assignments.getInteger("maxDepth"));
        rf.setBagSizePercent(assignments.getInteger("bagSizePercent"));
        rf.setNumIterations(assignments.getInteger("numIterations"));
        return rf;
    }
}
