package com.example;

import java.util.Arrays;
import java.util.Map;

import com.sigopt.Sigopt;
import com.sigopt.example.Franke;
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
                .name("Franke Optimization (Java)")
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
                .observationBudget(20)
                .build())
            .call();
        System.out.println("Created experiment: https://sigopt.com/experiment/" + experiment.getId());

        // Run the Optimization Loop between 10x - 20x the number of parameters
        for (int i = 0; i < experiment.getObservationBudget(); i++) {
            Suggestion suggestion = experiment.suggestions().create().call();
            Result result = App.evaluateModel(suggestion);
            Observation observation = experiment.observations().create()
                .data(new Observation.Builder()
                    .suggestion(suggestion.getId())
                    .value(result.value)
                    .valueStddev(result.stddev)
                    .build())
                .call();
        }
    }

    // Evaluate your model with the suggested parameter assignments
    // Franke function - http://www.sfu.ca/~ssurjano/franke2d.html
    public static Result evaluateModel(Suggestion suggestion) throws InterruptedException
    {
        Map<String, Object> assignments = suggestion.getAssignments();
        double x = (Double)assignments.get("x");
        double y = (Double)assignments.get("y");
        double result = Franke.evaluate(x, y);
        return new Result(result, null);
    }
}
