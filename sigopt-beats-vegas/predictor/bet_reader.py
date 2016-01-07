import BeautifulSoup
import datetime
import numpy
import pickle
import requests

from constant import SEASON_1415_START, SEASON_1516_END

def fix_game(game):
  fixes = {
    'Milwaukee': 'Bucks',
    'Trail': 'Trail Blazers',
    'Trailblazers': 'Trail Blazers',
    'Trail Blaze': 'Trail Blazers',
  }
  for key in ('home', 'away'):
    if key in game and game[key] in fixes:
      game[key] = fixes[game[key]]

def transform_old_format(old_bet_info):
  bet_info = []
  for day_str, games in old_bet_info.iteritems():
    for game in games:
      fix_game(game)
      bet_info.append({
        'home': game['home'],
        'away': game['away'],
        'overunder': game['overunder'],
        'date': datetime.datetime.strptime(day_str, '%Y-%m-%d') - datetime.timedelta(days=1),
        })
  return bet_info

def read_info(filename="bet_info.pkl", start_date=SEASON_1415_START, end_date=SEASON_1516_END):
  try:
    with open(filename, 'rb') as f:
      bet_info_s15 = pickle.load(f)
  except:
    print 'Downloading betting info from {} to {}.\nThis should only need to be done once...'.format(
      start_date,
      end_date,
    )

    bet_info = {}
    day = end_date
    while day >= start_date:
      print day
      if day.month == 7:
        print "Skipping Summer"
        day = day - datetime.timedelta(days=90)
        continue
      page_url = 'http://data.nowgoal.com/nba/oddsHistory.aspx?Selday={day}'.format(day=day)
      page = None
      for _ in range(5):
        try:
          page = requests.get(page_url, timeout=10)
        except:
          print "retrying..."
          continue
      if page == None:
        print "Failed to get bets for {} after 5 retries.".format(day)
        continue

      soup = BeautifulSoup.BeautifulSoup(page.text)
      rows = soup.findAll(id='Sclass_1') # Sclass_1 = NBA
      if len(rows) > 0:
        bet_info[day] = []
      for row in rows:
        if len(row) < 11:
          continue
        overunder = float(row.findAll("td")[10].find("a").contents[0]) # Bet365
        line = float(row.findAll("td")[5].find("a").contents[0]) # Bet365
        trows = row.find(align='left').findAll(target="_blank") # Team names
        assert len(trows) == 2
        home = trows[0].contents[0].strip()
        away = trows[1].contents[0].strip()
        bet_info[day].append({
            "home": home,
            "away": away,
            "overunder": overunder,
            "line": line
          })
      day = day - datetime.timedelta(days=1)

    bet_info_s15 = {}
    for day, info in bet_info.iteritems():
      if day >= start_date and day <= end_date:
        bet_info_s15[day.strftime('%Y-%m-%d')] = info

    with open(filename, "wb") as f:
      pickle.dump(bet_info_s15, f)
  return bet_info_s15
