__author__ = 'gt'

from gt.sbu.so.data import get_training_data, get_test_data, get_validation_data
from gt.sbu.so.ft_extraction import get_features
from sklearn import svm

def train(sents, labels):
  ft_extractor,X = get_features(sents)
  #cache for the kernel in MB
  # clf            = svm.SVC(kernel='rbf', cache_size=1024, C=1000.0)
  # clf = svm.SVC(C=0.10000000000000001, cache_size=200, class_weight=None, coef0=0.0,
  #               degree=3, gamma=0.01, kernel='rbf', max_iter=-1, probability=False,
  #               random_state=None, shrinking=True, tol=0.001, verbose=False
  # )

  #Tuned kernel for all words unigram, bigram
  clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
                random_state=None, shrinking=True, tol=0.001, verbose=False)
  # clf = svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  # gamma=1.0000000000000001e-05, kernel='linear', max_iter=-1,
  # probability=False, random_state=None, shrinking=True, tol=0.001,
  # verbose=False)

  clf.fit(X,labels)
  print ft_extractor.get_feature_names()
  return ft_extractor,clf

def getBestEstimator(X, labels):
  # from sklearn.preprocessing import Scaler
  import numpy as np
  from sklearn.grid_search import GridSearchCV
  from sklearn.cross_validation import StratifiedKFold

  # scaler = Scaler()
  # X = scaler.fit_transform(X)

  C_range = 10. ** np.arange(-2, 9)
  gamma_range = 10. ** np.arange(-5, 4)

  param_grid = dict(gamma=gamma_range, C=C_range)

  grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=StratifiedKFold(y=labels, k=5))

  grid.fit(X, labels)

  print("The best classifier is: ", grid.best_estimator_)


def test(sents, ft_extractor, clf):
  X = ft_extractor.transform(sents)
  y = clf.predict(X)
  return y

def evaluate(observed, expected):
  if len(observed) != len(expected):
    raise 'Number of observations != Number of experiments'

  ctr = 0
  tpr = 0
  fpr = 0

  for i in range(len(observed)):
    if observed[i] != expected[i] and observed[i] == '+':
      fpr += 1
    if observed[i] == expected[i]:
      if(observed[i] == '+'):
        tpr +=1
      ctr+=1

  tpr = tpr * 1.0
  fpr = fpr * 1.0
  tpr = tpr / (len([x for x in expected if x == '+']))
  fpr = fpr / (len([x for x in expected if x == '-']))
  print "TPR " + str(tpr)
  print "FPR " + str(fpr)
  print "Observed + " + str(len([x for x in observed if x == '+']))
  print "Observed - " + str(len([x for x in observed if x == '-']))
  return ctr



def run_classifier():
  # print 'getting training data...'
  sents, labels = get_training_data()
  # pprint(sents)
  # pprint(labels)
  print 'Training set size ' + str(len(labels))

  # print 'training on the data...'
  ft_xtractor, clf = train(sents, labels)

  print 'number of features: ' + str(len(ft_xtractor.get_feature_names()))

  # print 'getting test data...'
  valid_sents, expected_labels = get_validation_data()
  # pprint(test_sents)
  # pprint(expected_labels)
  print 'Testing set size ' + str(len(expected_labels))

  # print 'using the model to predict...'
  pred_labels = test(valid_sents, ft_xtractor, clf)
  correct = evaluate(pred_labels, expected_labels)

  print 'prediction accuracy...'
  print str( (correct * 100.0) / len(expected_labels))

def findEstimator():
  sents, labels = get_training_data()
  ft_extractor, X = get_features(sents)
  getBestEstimator(X,labels)


def main():
  run_classifier()


if __name__ == '__main__':
  import time
  start_time = time.time()
  main()
  print time.time() - start_time, "seconds"
  print '#############'
