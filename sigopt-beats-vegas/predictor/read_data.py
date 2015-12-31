import json
import datetime

from team_stats import TeamStats
from box_score_helpers import get_total_points_from_boxscore
from constant import TEAM_ID_TO_NAME

def read_box_scores(file_name='../boxscores/all_boxscores.json'):
  try:
    with open(file_name, 'r') as fp:
      box_scores = json.load(fp)
  except IOError:
    raise Exception('Could not open {} - make sure you have downloaded the boxscores.'.format(file_name))
  return box_scores

def get_teams(box_score):
  team_data = box_score['resultSets'][0]['rowSet'][0]
  home_team = TEAM_ID_TO_NAME[team_data[6]]
  away_team = TEAM_ID_TO_NAME[team_data[7]]
  return home_team, away_team

def generate_historical_games(box_scores, max_date=None):
  historical_games = []
  for game, box_score in box_scores.iteritems():
    sinfo = game.split('-')
    day_str = '{0}-{1}-{2}'.format(sinfo[0], sinfo[1], sinfo[2])
    date = datetime.datetime.strptime(day_str, '%Y-%m-%d')
    if max_date is not None:
      if date > max_date:
        continue
    home_team, away_team = get_teams(box_score)
    points = get_total_points_from_boxscore(box_score)
    if points is not None:
      historical_games.append({
        'home': home_team,
        'away': away_team,
        'total_score': points,
        'date': date,
        })
  return historical_games

def generate_all_stats(box_scores, all_stats=None):

  if not all_stats:
    all_stats = {}

  for game, box_score in box_scores.iteritems():
    sinfo = game.split('-')
    day_str = '{0}-{1}-{2}'.format(sinfo[0], sinfo[1], sinfo[2])
    date = datetime.datetime.strptime(day_str, '%Y-%m-%d')
    home_team, away_team = get_teams(box_score)

    if home_team not in all_stats:
      all_stats[home_team] = TeamStats(home_team)

    if away_team not in all_stats:
      all_stats[away_team] = TeamStats(away_team)

    all_stats[home_team].add_game_to_stats(box_score, True)
    all_stats[away_team].add_game_to_stats(box_score, False)

  return all_stats
