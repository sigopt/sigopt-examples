# sigopt-beats-vegas

Learn more at the associated [blog post](http://blog.sigopt.com/post/136340340198/sigopt-for-ml-using-model-tuning-to-beat-vegas).

Dive right in with [the iPython Notebook](https://github.com/sigopt/sigopt-examples/blob/master/sigopt-beats-vegas/SigOpt%20NBA%20OverUnder%20Model.ipynb).

#### Setup
```
sudo apt-get install libffi-dev python-pip python-dev libssl-dev libxml2-dev libxslt1-dev python-scipy git
git clone https://github.com/sigopt/optimal-hack-bet.git
cd optimal-hack-bet
sudo pip install -r requirements.txt
```

Next, open `predictor/sigopt_tokens.py` and add in your SigOpt auth tokens
(which can be found at https://sigopt.com/user/profile).

#### Fetching the data
```
cd boxscores/scraper
./scrape_all
```

#### Running
```
cd predictor
python stand_alone.py
```
