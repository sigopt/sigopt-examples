from sklearn import ensemble
import numpy
from joblib import Parallel, delayed
import os

import model
import evaluator

def joblib_wrapper(historical_games_trunc, all_stats, bet_info, historical_games_by_tuple, tunable_param_list):
  (moving_averages, transform_params, n_estimators, min_samples_split, min_samples_leaf, bet_threshold) = tunable_param_list

  print 'Building model...'
  X, y = model.build_model_inputs(historical_games_trunc, all_stats, moving_averages, transform_params)
  the_model = model.build_model(X, y, n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
  print 'Evaluating model...'
  winnings = evaluator.evaluate_model(the_model, all_stats, bet_info, historical_games_by_tuple, moving_averages, transform_params, bet_threshold)

  return winnings

def runner(historical_games_trunc, historical_games_by_tuple, bet_info, all_stats, tunable_param_lists):
  os.system("taskset -p 0xffffffff %d" % os.getpid()) # numpy (openBLAS) breaks this in ubuntu for joblib
  winnings_list = Parallel(n_jobs=-2)(delayed(joblib_wrapper)(historical_games_trunc, all_stats, bet_info, historical_games_by_tuple, tunable_param_list) for tunable_param_list in tunable_param_lists)
  return winnings_list
