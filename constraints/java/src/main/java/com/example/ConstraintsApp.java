// Use SigOpt to define an experiment with constraints and optimize a version of the Adjiman function
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
import com.sigopt.model.LinearConstraint;
import com.sigopt.model.ConstraintTerm;

public class ConstraintsApp
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
                .name("Adjiman Optimization with Constraints (Java)")
                .parameters(Arrays.asList(
                    new Parameter.Builder()
                        .name("x")
                        .type("double")
                        .bounds(new Bounds(-1.0, 2.0))
                        .build(),
                    new Parameter.Builder()
                        .name("y")
                        .type("double")
                        .bounds(new Bounds(-1.0, 1.0))
                        .build()
                ))
                .linearConstraints(java.util.Arrays.asList(
                    new LinearConstraint.Builder()
                        // Constraint equation: x + y >= 1
                        .terms(java.util.Arrays.asList(
                            new ConstraintTerm.Builder()
                                .name("x")
                                .weight(1)
                                .build(),
                            new ConstraintTerm.Builder()
                                .name("y")
                                .weight(1)
                                .build()
                        ))
                        .threshold(1)
                        .type("greater_than")
                        .build(),
                    new LinearConstraint.Builder()
                        // Constraint equation: x - y >= 1
                        .terms(java.util.Arrays.asList(
                            new ConstraintTerm.Builder()
                                .name("x")
                                .weight(1)
                                .build(),
                            new ConstraintTerm.Builder()
                                .name("y")
                                .weight(-1)
                                .build()
                        ))
                        .threshold(1)
                        .type("greater_than")
                        .build()

                ))
                .observationBudget(30)
                .build())
            .call();
        System.out.println("Created experiment: https://sigopt.com/experiment/" + experiment.getId());

        // Run the Optimization Loop between 10x - 20x the number of parameters
        for (int i = 0; i < experiment.getObservationBudget(); i++) {
            // Receive a Suggestion from SigOpt
            Suggestion suggestion = experiment.suggestions().create().call();

            // Evaluate the function
            double value = ConstraintsApp.adjimanFunction(suggestion.getAssignments());

            // Report an Observation (with standard deviation) back to SigOpt
            Observation observation = experiment.observations().create()
                .data(new Observation.Builder()
                    .suggestion(suggestion.getId())
                    .value(value)
                    .build())
                .call();
        }
    }

    // Constrained variation on the Adjiman Function http://benchmarkfcns.xyz/benchmarkfcns/adjimanfcn.html
    public static double adjimanFunction(Assignments assignments) throws java.lang.Exception
    {
        double x = assignments.getDouble("x");
        double y = assignments.getDouble("y");

        // Multiply by -1 because SigOpt maximizes functions
        return -1 * (Math.cos(x) * Math.sin(y) - x / (Math.pow(y, 2) + 1));
    }
}
