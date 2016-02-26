# SigOpt with Other Languages
Our `other_languages` example is one way to use SigOpt when your metric evaluation function is in a language other than python. All you need to do is create an executable file that accepts parameters as command line arguments, and then create an experiment with the same parameter names as the executable. The executable file should accept the suggested parameters at the command line, evaluate the metric, and print out a float.

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile).
3. `export CLIENT_TOKEN=<your client_token>`
4. Install the SigOpt python client `pip install sigopt`

## Example Usage
```
python other_languages.py --command='<command to run your script>' --experiment_id=EXPERIMENT_ID --client_token=$CLIENT_TOKEN
```
The above command will run the following sub process to evaluate your metric, automatially requesting the suggsetion beforehand and reporting the observation afterwards:
```
<comamnd to run your script> --x=SUGGESTION_FOR_X
```

Feel free to use, or modify for your own needs!



