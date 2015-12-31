import datetime

def get_datetime_from_boxscore(box_score):
  return datetime.datetime.strptime(
      box_score['resultSets'][0]['rowSet'][0][0].split('T')[0],
      '%Y-%m-%d',
      )

def get_total_points_from_boxscore(box_score):
  points_rows = box_score['resultSets'][1]['rowSet']
  if points_rows and len(points_rows) >= 2:
    assert len(points_rows) == 2
    home_points = points_rows[0][-1]
    away_points = points_rows[1][-1]
    if home_points is not None and away_points is not None:
      return home_points + away_points
    else:
      return None
  else:
    return None
