[![image](https://sigopt.com/static/img/SigOpt_logo_horiz.png?raw=true)](https://sigopt.com)

# Collaborative Filtering Tuning

Example using SigOpt and Spark / MLLib to tune an alternating least squares algorithm (ALS) for collaborative filtering.

## Setup

1. Get a free SigOpt account at [https://sigopt.com/signup](https://sigopt.com/signup)
2. Find your `client_token` on your [user profile](https://sigopt.com/user/profile) and set it
  as the `SIGOPT_API_TOKEN` environment variable.
3. `git clone https://github.com/sigopt/sigopt-examples.git`
4. `cd sigopt-examples/spark/recommender_sys`
5. `sudo ./setup_env.sh`   (Note this assumes you are running on master node of spark-ec2 cluster)
6. `sbt assembly`
7. ```
   ./spark/bin/spark-submit \
  --master spark://<YOUR_SPARK_MASTER_DNS>:7077 \
  --class "MovieLensExperiment" \
  target/scala-2.10/movie-lens-sigopt-assembly-1.0.jar   # Might be scala-2.11, depending on your version
  ```
