package com.example;

import java.util.Arrays;
import java.util.Map;

import com.sigopt.Sigopt;
import com.sigopt.exception.APIException;
import com.sigopt.model.Bounds;
import com.sigopt.model.Experiment;
import com.sigopt.model.Observation;
import com.sigopt.model.Parameter;
import com.sigopt.model.Suggestion;

class Result
{
    Double value;
    Double stddev;

    public Result(Double value, Double stddev)
    {
        this.value = value;
        this.stddev = stddev;
    }
}

public class App
{
    public static void main(String[] args) throws APIException, InterruptedException
    {
        String clientId = null;
        for (int i = 0; i < args.length; i++) {
            if (args[i].equals("--client_token") && i < args.length - 1) {
                Sigopt.clientToken = args[i+1];
            }
        }

        Experiment experiment = Experiment.create(
            new Experiment.Builder()
                .name("Franke Function")
                .parameters(Arrays.asList(
                    new Parameter.Builder()
                        .name("x")
                        .type("double")
                        .bounds(new Bounds(0.0, 1.0))
                        .build(),
                    new Parameter.Builder()
                        .name("y")
                        .type("double")
                        .bounds(new Bounds(0.0, 1.0))
                        .build()
                ))
                .build())
            .call();

        System.out.println("Follow your experiment results here: https://sigopt.com/experiment/" + experiment.getId());
        for (int i = 0; i < 20; i++) {
            Suggestion suggestion = experiment.suggestions().create().call();
            System.out.println("Computing result for suggestion: " + suggestion.toString());
            Result result = App.evaluateMetric(suggestion);
            Observation observation = experiment.observations().create()
                .data(new Observation.Builder()
                    .suggestion(suggestion.getId())
                    .value(result.value)
                    .valueStddev(result.stddev)
                    .build())
                .call();
            System.out.println("Reported observation: " + observation.toString());
        }
    }

    // This should produce the value you want to optimize - substitute in your own problem here.
    // Use the suggested values to compute your result
    public static Result evaluateMetric(Suggestion suggestion) throws InterruptedException
    {
        Map<String, Object> assignments = suggestion.getAssignments();
        double x = (Double)assignments.get("x");
        double y = (Double)assignments.get("y");

        // Franke function - http://www.sfu.ca/~ssurjano/franke2d.html
        double result = (
          0.75 * Math.exp(Math.pow(-(9 * x - 2), 2.0) / 4.0 - Math.pow((9 * y - 2), 2.0) / 4.0) +
          0.75 * Math.exp(Math.pow(-(9 * x + 1), 2.0) / 49.0 - (9 * y + 1) / 10.0) +
          0.5 * Math.exp(Math.pow(-(9 * x - 7), 2.0) / 4.0 - Math.pow((9 * y - 3), 2.0) / 4.0) -
          0.2 * Math.exp(Math.pow(-(9 * x - 4), 2.0) - Math.pow((9 * y - 7), 2.0))
        );
        // Note: SigOpt was designed to optimze time consuming and expensive processes like
        // tuning ML models, optimizing complex simulations, or running optimal A/B tests.
        // We simulate the evaluation taking 1000ms here to allow SigOpt time to find the best
        // possible suggestion. In practice this is not an issue when the underlying evaluation
        // is time consuming or expensive enough to warrant the use of SigOpt.
        Thread.sleep(1000);
        return new Result(result, null);
    }
}
