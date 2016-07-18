import math, numpy, datetime
import sklearn
import random
import scipy
import sigopt
import scipy.io
import sklearn.cluster
import xgboost as xgb
from skimage.util import view_as_windows
from skimage.color import rgb2gray
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.covariance import LedoitWolf

# load SVHN dataset
unlab_X = scipy.io.loadmat("extra_32x32.mat")['X'].astype('float64')
test_X = scipy.io.loadmat("test_32x32.mat")['X'].astype('float64')
test_y = scipy.io.loadmat("test_32x32.mat")['y'].astype('float64')
train_X = scipy.io.loadmat("train_32x32.mat")['X'].astype('float64')
train_y = scipy.io.loadmat("train_32x32.mat")['y'].astype('float64')

def convert_rgb2gray(X):
  X_gray = numpy.zeros((32, 32, X.shape[3]))
  for i in xrange(0, X.shape[3]):
    img_gray = rgb2gray(X[:, :, :, i])
    X_gray[:, :, i] = img_gray
  return X_gray

# convert all image data to grayscale
unlab_X = convert_rgb2gray(unlab_X)
test_X = convert_rgb2gray(test_X)
train_X = convert_rgb2gray(train_X)

# setup SigOpt experiment
conn = sigopt.Connection()
experiment = conn.experiments().create(
  name='SVHN Classifier',
  parameters=[
    {'name': 'filter_w',       'type': 'int', 'bounds': {'min': 7, 'max': 10}},
    {'name': 'slide_w',        'type': 'int', 'bounds': {'min': 2, 'max': 8}},
    {'name': 'km_n_clusters',  'type': 'int',    'bounds': {'min': 50, 'max': 500}},
    {'name': 'zca_eps',        'type': 'double', 'bounds': {'min': math.log(0.0001), 'max': math.log(100.0)}},
    {'name': 'sparse_p',       'type': 'double', 'bounds': {'min': 0.0, 'max': 100.0}},
    {'name': 'xgb_subsmple',   'type': 'double', 'bounds': {'min': 0.5, 'max': 1.0}},
    {'name': 'xgb_mx_depth',   'type': 'int', 'bounds': {'min': 3, 'max': 15}},
    {'name': 'xgb_num_est',    'type': 'int', 'bounds': {'min': 10, 'max': 100}},
    {'name': 'xgb_lr',         'type': 'double', 'bounds': {'min': math.log(0.0001), 'max': math.log(100.0)}},
  ],
  observation_budget=90,
)

# run optimization loop
for j in range(experiment.observation_budget):
  suggestion = conn.experiments(experiment.id).suggestions().create()
  params = suggestion.assignments
  print params
  w = int(params['filter_w'])              # SIGOPT param  (filter width in pixels)
  s = int(params['slide_w'])               # SIGOPT param  (held fixed at 2)
  q = params['sparse_p']                   # SIGOPT param  (percentile of active centroid distances)
  n_clust = int(params['km_n_clusters'])   # SIGOPT param  (num of centroids to learn)
  mesh_r  = 2                              # (what is the pooling res in pixels)
  zca_eps = math.exp(params['zca_eps'])

  # stack a random subset image patches, 125K
  X_unlab_patches = []
  random.seed(42)
  print "Gathering examples..."
  # Use subsample of 200K for k-means and covariance estimates
  for i in random.sample(range(0, unlab_X.shape[2]), 200000):
    patches = view_as_windows(unlab_X[:, :, i], (w, w), step=s)
    re_shaped = numpy.reshape(patches, (patches.shape[0]*patches.shape[0], w * w))
    # normalize the patches, per sample
    re_shaped = preprocessing.scale(re_shaped, axis=1)
    X_unlab_patches.append(re_shaped)
  X_unlab_patches = numpy.vstack(X_unlab_patches)

  # build whitening transform matrix
  print "Fitting ZCA Whitening Transform..."
  cov = LedoitWolf()
  cov.fit(X_unlab_patches)  # fit covariance estimate
  D, U = numpy.linalg.eigh(cov.covariance_)
  V = numpy.sqrt(numpy.linalg.inv(numpy.diag(D + zca_eps)))
  Wh = numpy.dot(numpy.dot(U, V), U.T)
  mu = numpy.mean(X_unlab_patches, axis=0)
  X_unlab_patches = numpy.dot(X_unlab_patches-mu, Wh)

  # run k-means on unlabelled data
  print "Starting k-means..."
  clustr = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clust,
                                           compute_labels=False,
                                           batch_size=300)
  k_means = clustr.fit(X_unlab_patches)


  def f_unsup(img):
    img_ptchs = view_as_windows(img, (w, w), step=s)
    orig_dim = (img_ptchs.shape[0], img_ptchs.shape[1], k_means.cluster_centers_.shape[0])
    img_ptchs = numpy.reshape(img_ptchs, (img_ptchs.shape[0] * img_ptchs.shape[0], w * w))
    # normalize the patches, per patch sample
    img_ptchs = preprocessing.scale(img_ptchs, axis=1)
    # apply whitening transform
    img_ptchs = numpy.dot(img_ptchs - mu, Wh)   # uses multiprocessing!
    Z = k_means.transform(img_ptchs)
    # sparsity constraint threshold q
    uZ = numpy.percentile(Z, q, axis=1, keepdims=True)
    tri_Z = numpy.maximum(0, uZ - Z)
    tri_Z_image = numpy.reshape(tri_Z, orig_dim)

    # pooling of averages at mesh resolution
    pooled_Z = []
    pool_dim1 = orig_dim[0]/mesh_r or orig_dim[0]
    pool_dim2 = orig_dim[1]/mesh_r or orig_dim[1]
    for i in xrange(0,orig_dim[0], pool_dim1):
      for j in xrange(0,orig_dim[1], pool_dim2):
        POOL_MSH = numpy.mean(tri_Z_image[i:(i + pool_dim1), j:(j + pool_dim2)], axis=(0, 1))
        pooled_Z.append(POOL_MSH)
    return numpy.hstack(pooled_Z)

  # helper function to transform image data
  def process_chunk(lower, upper, X):
    res_Z = []
    for i in xrange(lower, min(upper, X.shape[2])):
      Z = f_unsup(X[:, :, i])
      res_Z.append(Z)
    return res_Z

  print "Transforming Training Data..."
  train_XZ = process_chunk(0, train_X.shape[2], train_X)
  train_XZ = numpy.vstack(train_XZ)
  train_XZ_2, valid_XZ, train_y_2, valid_y = train_test_split(train_XZ, train_y,
                                                              test_size=0.25, random_state=42)

  print "Transforming Test Data..."
  test_XZ = process_chunk(0, test_X.shape[2], test_X)
  test_XZ = numpy.vstack(test_XZ)

  clf = xgb.XGBClassifier(max_depth=int(params['xgb_mx_depth']), n_estimators=int(params['xgb_num_est']),
                          learning_rate=math.exp(params['xgb_lr']), subsample=params['xgb_subsmple'])

  print "Training Classifier..."
  clf.fit(train_XZ_2, train_y_2.ravel())

  # Cross validation metric: SigOpt optimizes this metric
  y_valid_pred = clf.predict(valid_XZ)
  opt_metric = accuracy_score(valid_y, y_valid_pred)

  # Hold out metric: generalization error of final params on never trained on test set
  # y_test_pred = clf.predict(test_XZ)
  # opt_metric = accuracy_score(test_y, y_test_pred)
  # print opt_metric

  print "DONE!"
  conn.experiments(experiment.id).observations().create(
    suggestion=suggestion.id,
    value=opt_metric,
    value_stddev=0.03
  )

