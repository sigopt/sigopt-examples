# SigOpt with Other Languages
If your metric evaluation is in an executable file written in another language you can use the example `other_languages` as a starting point to perform metric evaluation in a sub process.

This script will pass the suggested assignments as command line arguments, and will expect the script's output to be a float representing a fuction evaluated on these assignments.

To start, you'll need to create an experiment with parameter names matching the command line arguments of your program.

## Example Usage

For example, your filename is `test` and it expects an argument `x` on the command line that is a double. Say this script normally spews setup info, but if you run `./test --quiet`, the only information sent to stdout is the value of your function evalauted at `x`. Set up an experiment with one parameter named `x` that has type `double`, and run the following command:
```
python other_languages.py --command='./test --quiet' --experiment_id=EXPERIMENT_ID --client_token=$CLIENT_TOKEN
```
The above command will run the following sub process to evaluate your metric, automatially requesting the suggsetion beforehand and reporting the observation afterwards:
```
./test --quiet --x=SUGGESTION_FOR_X
```

Feel free to use, or modify for your own needs!



