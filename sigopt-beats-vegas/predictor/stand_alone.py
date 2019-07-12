import sigopt

import datetime, os

import bet_reader
import evaluator
import read_data
from constant import SEASON_1314_END
from game_stats import EXP_TRANSFORM
from run_model import runner

def create_sigopt_experiment(conn, sigopt_depth):
  """Creates and returns a SigOpt experiment object."""
  experiment = conn.experiments().create(
    name='NBA Over/Under',
    project='sigopt-examples',
    metrics=[dict(name='winnings', objective='maximize')],
    parameters=[
              {'name': 'fast_ma',
               'type': 'int',
               'bounds': { 'min': 1, 'max': 5 },
              },
              {'name': 'slow_ma',
               'type': 'int',
               'bounds': { 'min': 6, 'max': 10 },
              },
              {'name': 'exp_param',
               'type': 'double',
               'bounds': { 'min': 0.0, 'max': 1.0 },
              },
              {'name': 'n_estimators',
               'type': 'int',
               'bounds': { 'min': 100, 'max': 2000 },
              },
              {'name': 'min_samples_split',
               'type': 'int',
               'bounds': { 'min': 2, 'max': 10 },
              },
              {'name': 'min_samples_leaf',
               'type': 'int',
               'bounds': { 'min': 1, 'max': 10 },
              },
              {'name': 'bet_threshold',
               'type': 'double',
               'bounds': { 'min': 0, 'max': 5 },
              },
          ],
    observation_budget=sigopt_depth,
  )

  print "You can track your experiment at https://sigopt.com/experiment/{0}".format(experiment.id)

  return experiment

def get_historical_games(box_scores, max_date=None):
  all_stats = read_data.generate_all_stats(box_scores)
  historical_games = read_data.generate_historical_games(
      box_scores,
      max_date=max_date,
  )

  return historical_games

def run_sigopt(box_scores, historical_games, historical_games_training_set, bet_info, sigopt_width=1, sigopt_depth=100):
  historical_games_by_tuple = evaluator.get_historical_games_by_tuple(historical_games)

  conn = sigopt.Connection()
  experiment = create_sigopt_experiment(conn, sigopt_depth)

  for _ in range(experiment.observation_budget):
      tunable_param_lists = []
      suggestion_ids = []
      for worker_id in range(sigopt_width):
          conn = sigopt.Connection()
          suggestion = conn.experiments(experiment.id).suggestions().create()
          suggestion_ids.append(suggestion.id)

          moving_averages = (
              suggestion.assignments['slow_ma'],
              suggestion.assignments['fast_ma'],
              )

          transform_params = {
              'type': EXP_TRANSFORM,
              'exp_param': suggestion.assignments['exp_param'],
              }

          tunable_param_lists.append([
            moving_averages,
            transform_params,
            suggestion.assignments['n_estimators'],
            suggestion.assignments['min_samples_split'],
            suggestion.assignments['min_samples_leaf'],
            suggestion.assignments['bet_threshold'],
            ])

      all_stats = read_data.generate_all_stats(box_scores)
      winnings_list = runner(
          historical_games_training_set,
          historical_games_by_tuple,
          bet_info,
          all_stats,
          tunable_param_lists,
          )

      for i, suggestion_id in enumerate(suggestion_ids):

          conn.experiments(experiment.id).observations().create(
            suggestion=suggestion_id,
            value=winnings_list[i][0],
            value_stddev=winnings_list[i][1],
          )

  print "Optimization done. View results at https://sigopt.com/experiment/{0}".format(experiment.id)

  experiment_detail = conn.experiments(experiment.id).fetch()
  best_observation = experiment_detail.progress.best_observation
  if best_observation:
    print "Best value found: {0} at {1}".format(best_observation.value, best_observation.assignments)

  return experiment.id

def run_example(sigopt_width=1, sigopt_depth=100):
  boxscores_path = os.path.join(os.path.dirname(__file__), '../boxscores/all_boxscores.json')
  box_scores = read_data.read_box_scores(boxscores_path)

  historical_games = get_historical_games(box_scores)
  historical_games_training_set = get_historical_games(box_scores, max_date=SEASON_1314_END)

  bet_info_s15 = bet_reader.read_info()
  bet_info = bet_reader.transform_old_format(bet_info_s15)
  historical_games_by_tuple = evaluator.get_historical_games_by_tuple(historical_games)

  return run_sigopt(box_scores, historical_games, historical_games_training_set, bet_info, sigopt_width=sigopt_width, sigopt_depth=sigopt_depth)

if __name__ == '__main__':
  run_example()
