from collections import namedtuple

FeatureSet = namedtuple(
    "FeatureSet",
    [
      "PTSpm", # Points per minute
      "OREBpm", # Offensive rebounds per minute
      "DREBpm", # Defensive rebounds per minute
      "STLpm", # Steals per minute
      "BLKpm", # Blocks per minute
      "ASTpm", # Assists per minute
      "PIPpm", # Points in paint per minute
      "SCPpm", # Second chance points per minute
      "FBPpm", # Fast break points per minute
      "LCpm", # Lead changes per minute
      "TTpm", # Times tied per minute
      "LLpg", # Largest lead per game
      "PDIFFpg", # Point differential per game
      "FGApm", # Field goal attempts per minute
      "FGMpm", # Field goals made per minute
      "FTApm", # Free throw attempts per minute
      "FTMpm", # Free throws made per minute
      "TPFGApm", # Three point field goals attempted per minute
      "TPFGMpm", # Three point field goals made per minute
      "Q1PTSpg", # First quarter points per game
      "Q2PTSpg", # Second quarter points per game
      "Q3PTSpg", # Third quarter points per game
      "Q4PTSpg", # Fourth quarter points per game
      ],
    )

def calculate_features_from_boxscore(box_score, is_home_game):
  """Returns a FeatureSet given stats and a team_id.

  stats - a list of result sets from a game
  team_id - 0 for home, 1 for away
  """
  if is_home_game:
    team_id = 0
  else:
    team_id = 1
  stats = box_score['resultSets']
  try:
    time = stats[5]['rowSet'][team_id][5]
    if time:
      minutes = float(time.split(':')[0])
      return FeatureSet(
        int(stats[5]['rowSet'][team_id][23]) / minutes,
        int(stats[5]['rowSet'][team_id][15]) / minutes,
        int(stats[5]['rowSet'][team_id][16]) / minutes,
        int(stats[5]['rowSet'][team_id][19]) / minutes,
        int(stats[5]['rowSet'][team_id][20]) / minutes,
        int(stats[5]['rowSet'][team_id][18]) / minutes,
        int(stats[6]['rowSet'][team_id][5]) / minutes,
        int(stats[6]['rowSet'][team_id][6]) / minutes,
        int(stats[6]['rowSet'][team_id][7]) / minutes,
        int(stats[6]['rowSet'][team_id][9]) / minutes,
        int(stats[6]['rowSet'][team_id][10]) / minutes,
        int(stats[6]['rowSet'][team_id][9]),
        int(stats[5]['rowSet'][team_id][24]),
        int(stats[5]['rowSet'][team_id][7]) / minutes,
        int(stats[5]['rowSet'][team_id][6]) / minutes,
        int(stats[5]['rowSet'][team_id][13]) / minutes,
        int(stats[5]['rowSet'][team_id][12]) / minutes,
        int(stats[5]['rowSet'][team_id][10]) / minutes,
        int(stats[5]['rowSet'][team_id][9]) / minutes,
        int(stats[1]['rowSet'][team_id][7]),
        int(stats[1]['rowSet'][team_id][8]),
        int(stats[1]['rowSet'][team_id][9]),
        int(stats[1]['rowSet'][team_id][10]),
        )
    else:
      return None
  except IndexError:
    # Missing some crucial data, so skip this record
    return None
