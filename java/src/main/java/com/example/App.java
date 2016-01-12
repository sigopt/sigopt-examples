package com.example;

import java.util.Map;

import com.sigopt.Sigopt;
import com.sigopt.exception.APIException;
import com.sigopt.model.Experiment;
import com.sigopt.model.Observation;
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
    public static void main(String[] args) throws APIException
    {
        // Fill these in with your credentials
        Sigopt.clientToken = "client_token";
        Sigopt.userToken = "user_token";

        // Fill this in with your experiment
        String EXPERIMENT_ID = "4";

        Experiment experiment = Experiment.retrieve(EXPERIMENT_ID).call();
        while (true) {
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
        }
    }

    public static Result computeResult(Suggestion suggestion)
    {
        Map<String, Object> assignments = suggestion.getAssignments();
        // Use the suggested values to compute your result
        // Example: ((Double)assignments.get("x")) * 5 + ((Double)assignments.get("y")) * 3
        return new Result(1.0, null);
    }
}
