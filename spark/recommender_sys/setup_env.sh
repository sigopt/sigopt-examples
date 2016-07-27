#!/bin/bash
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
unzip ml-latest.zip
/root/ephemeral-hdfs/bin/hadoop fs -put ml-latest/ratings.csv data/ml-latest/ratings.csv
/root/ephemeral-hdfs/bin/hadoop fs -put ml-latest/movies.csv data/ml-latest/movies.csv
