# Classifier Tuning

Machine learning classifier hyperparameter optimization example.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models).

Setup:

1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your SigOpt API tokens and client ID at https://sigopt.com/user/profile
3. Install requirements `pip install -r requirements.txt`

Running Examples:

```bash
# Run default example using small sklearn dataset and Gradient Boosting Classifier
python classifier_tuner.py --user-token USER_TOKEN \
  --client-token CLIENT_TOKEN --client-id CLIENT_ID

# Run using connect-4 dataset (this takes a long time) and Support Vector Classfier
python classifier_tuner.py --user-token USER_TOKEN \
  --client-token CLIENT_TOKEN --client-id CLIENT_ID \
  --classifier-type SVC --dataset-name connect-4 --test-set-size 7557

# See full options
python classifier_tuner.py --help

optional arguments:
  -h, --help            show this help message and exit
  --user-token USER_TOKEN
                        Your SigOpt API user token
  --client-token CLIENT_TOKEN
                        Your SigOpt API client token
  --client-id CLIENT_ID
                        Your SigOpt API client id
  --classifier-type {GBC,SVC}
                        The type of classifier to use. Defaults to GBC
  --dataset-name DATASET_NAME
                        The sklearn dataset to use. Defaults to
                        datasets.load_digits().
  --test-set-size TEST_SET_SIZE
                        The number of points in the test set. The remainder of
                        the dataset will be the test set.
  --num-sigopt-suggestions NUM_SIGOPT_SUGGESTIONS
                        The number of suggestions to request from SigOpt.
  --grid-search-width GRID_SEARCH_WIDTH
                        How many grid points in each dimension to use for grid
                        search
  --num-random-searches NUM_RANDOM_SEARCHES
                        How many random search parameter configurations to
                        test
  --delete-existing     Delete all your SigOpt experiments and start again.
                        WARNING: This cannot be undone.
```

If you have any questions, comments, or concerns please email us at contact@sigopt.com
