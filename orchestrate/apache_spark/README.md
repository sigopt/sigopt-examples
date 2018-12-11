# Apache Spark with Sigopt Orchestrate

This is an example showing how to start a standalone Apache Spark cluster with one master node and one slave node within a SigOpt Orchestrate experiment.
This example uses Apache Spark 2.4.0 but you can replace this with the version you'd like to use by editing the
[orchestrate.yml](orchestrate.yml) for this experiment.

## Instructions to run this example

1. `git clone https://github.com/sigopt/sigopt-examples.git`
2. `cd sigopt-examples/orchestrate`
3. `sigopt cluster create -f clusters/i3_cluster.yml` This creates an i3 cluster but this example can work with CPU or GPU machines as well
4. `sigopt run --directory apache_spark -f apache_spark/orchestrate.yml`

> If you have any questions, please email us at support@sigopt.com or open a Github issue on this repository and someone from our team
will get back to you.
