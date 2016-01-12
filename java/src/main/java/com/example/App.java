package com.example;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.Map;

import com.sigopt.Sigopt;
import com.sigopt.exception.APIException;
import com.sigopt.model.Bounds;
import com.sigopt.model.Experiment;
import com.sigopt.model.Observation;
import com.sigopt.model.Parameter;
import com.sigopt.model.SuggestData;
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
            if (args[i].equals("--user_token") && i < args.length - 1) {
                Sigopt.userToken = args[i+1];
            }
            if (args[i].equals("--client_id") && i < args.length - 1) {
                clientId = args[i+1];
            }
        }

        if (clientId == null) {
            throw new RuntimeException("No client_id provided.");
        }


        Experiment experiment = Experiment.create(
                new Experiment.Builder()
                    .name("EggHolder Function")
                    .parameters(new ArrayList<Parameter>(Arrays.asList(new Parameter[]{
                        new Parameter.Builder()
                            .name("x1")
                            .type("double")
                            .bounds(new Bounds(-100.0, 100.0))
                            .build(),
                        new Parameter.Builder()
                            .name("x2")
                            .type("double")
                            .bounds(new Bounds(-100.0, 100.0))
                            .build()
                    })))
                    .build(),
                clientId)
            .call();

        for (int i = 0; i < 20; i++) {
            Suggestion suggestion = experiment.suggest().call();
            System.out.println("Computing result for suggestion: " + suggestion.toString());
            Result result = App.computeResult(suggestion);
            experiment.report()
                .addParam("data", new Observation.Builder()
                    .assignments(suggestion.getAssignments())
                    .value(result.value)
                    .valueStddev(result.stddev)
                    .build())
                .call();
            Thread.sleep(1000);
        }
        System.out.println("Check out your experiment results here: https://sigopt.com/experiment/" + experiment.getId());
    }

    // This should produce the value you want to optimize - substitute in your own problem here.
    // Use the suggested values to compute your result
    public static Result computeResult(Suggestion suggestion)
    {
        Map<String, Object> assignments = suggestion.getAssignments();
        double x1 = (Double)assignments.get("x1");
        double x2 = (Double)assignments.get("x2");
        double result = (
          (x2 + 47) * Math.sin(Math.sqrt(Math.abs(x2 + x1 / 2 + 47))) -
          x1 * Math.sin(Math.sqrt(Math.abs(x1 - (x2 + 47))))
        );
        return new Result(result, null);
    }
}
