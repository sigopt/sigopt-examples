import datetime
from bet_reader import fix_game

def build_history(bst, bet_info_s15):
  history = {}
  missing = []
  num_games = 0
  for day, games in bet_info_s15.iteritems():
    real_day = (datetime.datetime.strptime(day, '%Y-%m-%d') - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    history[day] = {}
    for game in games:
      fix_game(game)
      for bst_game in bst[real_day]:
        if game['home'] == bst_game['home'] and game['away'] == bst_game['away']:
          score = bst_game['stats']['resultSets'][1]['rowSet'][0][-1] + bst_game['stats']['resultSets'][1]['rowSet'][1][-1]
          history[day][tuple((game['home'], game['away']))] = score
          num_games += 1
  return history

def replay_history(bet_info, estimates, history, start='2014-10-28', end='2015-04-15'):
  winnings = 0
  num_games = 0
  wins = 0
  dtday = datetime.datetime.strptime(start, '%Y-%m-%d')
  end_day = datetime.datetime.strptime(end, '%Y-%m-%d')
  largest_up = 0
  largest_down = 0
  while dtday <= end_day:
    day = dtday.strftime('%Y-%m-%d')
    if day in bet_info:
      for game in bet_info[day]:
        team_tuple = tuple((game['home'], game['away']))
        if team_tuple not in history[day]:
          continue
        est = estimates[day][team_tuple]
        num_games += 1
        hist = history[day][team_tuple]
        overunder = game['overunder']
        if hist > overunder and est > overunder:
          # correct over bet
          winnings += 100
          wins += 1
        elif hist < overunder and est < overunder:
          # correct under bet
          winnings += 100
          wins += 1
        else:
          # loss
          winnings -= 110
        if winnings < largest_down:
          largest_down = winnings
        if winnings > largest_up:
          largest_up = winnings
    dtday += datetime.timedelta(days=1)
  print "Won {winnings} over {num_games} games. {wp}".format(
    winnings=winnings,
    num_games=num_games,
    wp=wins/float(num_games)
  )
  return winnings
