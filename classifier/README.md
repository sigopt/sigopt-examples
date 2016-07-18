# Classifier Tuning

Machine learning classifier hyperparameter optimization example.

More details about this example can be found in [the associated blog post](http://blog.sigopt.com/post/111903668663/tuning-machine-learning-models).

## Setup
1. Get a free SigOpt account at https://sigopt.com/signup
2. Find your `API_TOKEN` on your [user profile](https://sigopt.com/user/profile).
3. Install requirements `pip install -r requirements.txt`

## Run

Run default example using small sklearn dataset and Gradient Boosting Classifier.

```bash
python classifier_tuner.py --client-token API_TOKEN
```

Run using connect-4 dataset (this takes a long time) and Support Vector Classfier

```bash
python classifier_tuner.py --classifier-type SVC --dataset-name connect-4 --test-set-size 7557 --client-token API_TOKEN
```

If you have any questions, comments, or concerns please email us at contact@sigopt.com
