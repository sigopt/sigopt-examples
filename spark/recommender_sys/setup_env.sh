#!/bin/bash
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip
unzip ml-latest.zip
ephemeral-hdfs/bin/hadoop fs -put ml-latest/ratings.csv data/ml-latest/ratings.csv
ephemeral-hdfs/bin/hadoop fs -put ml-latest/movies.csv data/ml-latest/movies.csv
