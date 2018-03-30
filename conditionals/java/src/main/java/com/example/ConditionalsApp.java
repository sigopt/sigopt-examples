// Use SigOpt to define an experiment with conditionals and optimize a multivariate Gaussian distribution
// Learn more about SigOpt's Java Client:
// https://sigopt.com/docs/overview/java

package com.example;

import java.lang.Math;

import java.util.Arrays;

import com.sigopt.Sigopt;
import com.sigopt.exception.APIException;
import com.sigopt.model.Assignments;
import com.sigopt.model.Bounds;
import com.sigopt.model.Experiment;
import com.sigopt.model.Observation;
import com.sigopt.model.Parameter;
import com.sigopt.model.Suggestion;
import com.sigopt.model.Conditional;
import com.sigopt.model.Conditions;

public class ConditionalsApp
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


        // Create a SigOpt Experiment
        Experiment experiment = Experiment.create(
            new Experiment.Builder()
                .name("Multivariate Gaussian distribution Optimization with Conditionals (Java)")
                .parameters(Arrays.asList(
                    new Parameter.Builder()
                        .name("x")
                        .type("double")
                        .bounds(new Bounds(0.0, 1.0))
                        .conditions(
                            new Conditions.Builder()
                                .set("gaussian", java.util.Arrays.asList("gaussian1"))
                                .build()
                        )
                        .build(),
                    new Parameter.Builder()
                        .name("y")
                        .type("double")
                        .bounds(new Bounds(0.0, 1.0))
                        .conditions(
                            new Conditions.Builder()
                                .set("gaussian", java.util.Arrays.asList("gaussian2"))
                                .build()
                        )
                        .build(),
                    new Parameter.Builder()
                        .name("z")
                        .type("double")
                        .bounds(new Bounds(0.0, 1.0))
                        .conditions(
                            new Conditions.Builder()
                                .set("gaussian", java.util.Arrays.asList("gaussian1", "gaussian2"))
                                .build()
                        )
                        .build()
                ))
                .conditionals(java.util.Arrays.asList(
                    new Conditional.Builder()
                        .name("gaussian")
                        .values(java.util.Arrays.asList("gaussian1", "gaussian2"))
                        .build()
                ))
                .observationBudget(90)
                .build())
            .call();
        System.out.println("Created experiment: https://sigopt.com/experiment/" + experiment.getId());

        // Run the Optimization Loop between 10x - 20x the number of parameters
        for (int i = 0; i < experiment.getObservationBudget(); i++) {
            // Receive a Suggestion from SigOpt
            Suggestion suggestion = experiment.suggestions().create().call();

            // Evaluate the function
            double value = ConditionalsApp.multivariateGaussianDistribution(suggestion.getAssignments());

            // Report an Observation (with standard deviation) back to SigOpt
            Observation observation = experiment.observations().create()
                .data(new Observation.Builder()
                    .suggestion(suggestion.getId())
                    .value(value)
                    .build())
                .call();
        }
    }

    // Multivariate Gaussian distribution https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    public static double multivariateGaussianDistribution(Assignments assignments)
    {
        String gaussian = assignments.getString("gaussian");
        if (gaussian.equals("gaussian1")) {
            double x = assignments.getDouble("x");
            double z = assignments.getDouble("z");
            return (.5 * Math.exp(-10 * (.8 * Math.pow((x - .2), 2) + .7 * Math.pow((z - .5), 2))));
        } else { // gaussian is "gaussian2"
            double y = assignments.getDouble("y");
            double z = assignments.getDouble("z");
            return (.5 * Math.exp(-10 * (.7 * Math.pow((y - .4), 2) + .3 * Math.pow((z - .7), 2))));
        }
    }
}
