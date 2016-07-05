import json, math, numpy
import sigopt
from sigopt_creds import client_token
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation

#load text training data
POSITIVE_TEXT = json.load(open("POSITIVE_list.json"))
NEGATIVE_TEXT = json.load(open("NEGATIVE_list.json"))

# optimization metric : see blogpost http://blog.sigopt.com/post/133089144983/sigopt-for-ml-automatically-tuning-text
def sentiment_metric(POS_TEXT, NEG_TEXT, params):
    min_ngram = params['min_n_gram']
    max_ngram = min_ngram + params['n_gram_offset']
    min_doc_freq = math.exp(params['log_min_df'])
    max_doc_freq = min_doc_freq + params['df_offset']
    vectorizer = CountVectorizer(min_df=min_doc_freq, max_df=max_doc_freq,
                                 ngram_range=(min_ngram, max_ngram))
    X = vectorizer.fit_transform(POS_TEXT+NEG_TEXT)
    y = [1]*len(POS_TEXT) + [-1]*len(NEG_TEXT)
    clf = SGDClassifier(loss='log', penalty='elasticnet',
                        alpha=math.exp(params['log_reg_coef']), l1_ratio=params['l1_coef'])
    cv = cross_validation.ShuffleSplit(X.shape[0], n_iter=5, test_size=0.3, random_state=0)
    cv_scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
    return numpy.mean(cv_scores)

conn = sigopt.Connection(client_token=client_token)
experiment = conn.experiments().create(
  name='Sentiment LR Classifier',
  parameters=[
    { 'name':'l1_coef',      'type': 'double', 'bounds': { 'min': 0, 'max': 1.0 }},
    { 'name':'log_reg_coef', 'type': 'double', 'bounds': { 'min': math.log(0.000001), 'max': math.log(100.0) }},
    { 'name':'min_n_gram',   'type': 'int',    'bounds': { 'min': 1, 'max': 2 }},
    { 'name':'n_gram_offset','type': 'int',    'bounds': { 'min': 0, 'max': 2 }},
    { 'name':'log_min_df',   'type': 'double', 'bounds': { 'min': math.log(0.00000001), 'max': math.log(0.1) }},
    { 'name':'df_offset',    'type': 'double', 'bounds': { 'min': 0.01, 'max': 0.25 }}
  ],
  observation_budget=60,
)

# run experimentation loop
for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()
    opt_metric = sentiment_metric(POSITIVE_TEXT, NEGATIVE_TEXT, suggestion.assignments)
    conn.experiments(experiment.id).observations().create(
        suggestion=suggestion.id,
        value=opt_metric,
    ) # track progress on your experiment : https://sigopt.com/experiment/list
