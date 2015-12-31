import datetime
import numpy

import box_score_helpers
import features
from features import FeatureSet

MAX_LOOKBACK_DAYS = 90
LINEAR_TRANSFORM = 'linear_transform'
EXP_TRANSFORM = 'exp_transform'

class GameStats(object):
  """A Class for storing stats about specific games."""

  def __init__(self):
    self.stats_by_date = {}

  def add_game_to_stats(self, box_score, is_home_game):
    day_of_game = box_score_helpers.get_datetime_from_boxscore(box_score).date()
    stats = features.calculate_features_from_boxscore(box_score, is_home_game)
    if stats:
      self.stats_by_date[day_of_game] = stats

  @staticmethod
  def get_game_weight(game_num, total_num_games, transform_params):
    if transform_params == None:
      return 1.0
    elif transform_params['type'] == LINEAR_TRANSFORM:
      return (total_num_games - game_num) / float(total_num_games)
    elif transform_params['type'] == EXP_TRANSFORM:
      return numpy.exp(-1.0 * game_num * transform_params['exp_param'])
    else:
      raise NotImplementedError("Transform type {0} not implemented.".format(transform_params['type']))

  def get_average_stats_from_last_games(self, num_games, current_date, transform_params):
    current_date = current_date.date()
    stat_date = current_date - datetime.timedelta(days=1)
    sum_of_stats = numpy.zeros(len(FeatureSet._fields))
    num_games_averaged = 0
    while num_games_averaged < num_games:
      if stat_date in self.stats_by_date:
        weight = self.get_game_weight(num_games_averaged, num_games, transform_params)
        sum_of_stats += weight * numpy.array(self.stats_by_date[stat_date])
        num_games_averaged += 1
      stat_date -= datetime.timedelta(days=1)
      if stat_date < current_date - datetime.timedelta(days=MAX_LOOKBACK_DAYS):
        break

    if num_games_averaged < num_games:
      raise ValueError("Not enough previous games to average within MAX_LOOKBACK_DAYS ({0}).".format(MAX_LOOKBACK_DAYS))

    return sum_of_stats / float(num_games)
