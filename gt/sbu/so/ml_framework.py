__author__ = 'gt'

from gt.sbu.so.data \
  import get_training_data, get_validation_data, get_tsp_test_data

from gt.sbu.so.ft_extraction import get_features
from sklearn import svm
from sklearn.externals import joblib
from pprint import pprint
import math
import gt.sbu.so.tsp_adapter.tsp_instance as tsp

def train(sents, labels):
  ft_extractor,X = get_features(sents)
  #cache for the kernel in MB
  # clf            = svm.SVC(kernel='rbf', cache_size=1024, C=1000.0)
  # clf = svm.SVC(C=0.10000000000000001, cache_size=200, class_weight=None, coef0=0.0,
  #               degree=3, gamma=0.01, kernel='rbf', max_iter=-1, probability=False,
  #               random_state=None, shrinking=True, tol=0.001, verbose=False
  # )

  #Tuned kernel for all words unigram, bigram
  # clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  #               gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
  #               random_state=None, shrinking=True, tol=0.001, verbose=False)
  clf = svm.SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  gamma=1.0000000000000001e-05, kernel='linear', max_iter=-1,
  probability=True, random_state=None, shrinking=True, tol=0.001,
  verbose=False)

  clf.fit(X,labels)
  # print ft_extractor.get_feature_names()
  return ft_extractor,clf

def getBestEstimator(X, labels):
  # from sklearn.preprocessing import Scaler
  import numpy as np
  from sklearn.grid_search import GridSearchCV
  from sklearn.cross_validation import StratifiedKFold

  # scaler = Scaler()
  # X = scaler.fit_transform(X)

  C_range = 10. ** np.arange(-2, 9)
  # gamma_range = 10. ** np.arange(-5, 4)

  # param_grid = dict(gamma=gamma_range, C=C_range)
  param_grid = dict(C=C_range)
  grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid=param_grid, cv=StratifiedKFold(y=labels, k=5))

  grid.fit(X, labels)

  print("The best classifier is: ", grid.best_estimator_)


def test(sents, ft_extractor, clf):
  if len(sents) <= 1:
    print 'here'
  X = ft_extractor.transform(sents)

  y = clf.predict(X)
  prob = clf.predict_proba(X)

  # return y
  return prob,y

def evaluate(observed, expected, test_sents):
  from gt.sbu.so.data import blockSeparator
  if len(observed) != len(expected):
    raise 'Number of observations != Number of experiments'

  ctr = 0
  tpr = 0
  fpr = 0

  for i in range(len(observed)):
    # if observed[i] != expected[i]:
      # print '************'
      # print 'expected ' + str(expected[i])
      # print test_sents[i].split(blockSeparator)[0]
      # print test_sents[i].split(blockSeparator)[1]
      # print '************'
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
  ft_extractor = joblib.load('ft_xtractor_moretrainSamples.pkl')

  X = ft_extractor.transform(sents)

  getBestEstimator(X,labels)

def train_and_save():
  # print 'getting training data...'
  sents, labels = get_training_data()
  # pprint(sents)
  # pprint(labels)
  print 'Training set size ' + str(len(labels))

  # print 'training on the data...'
  ft_xtractor, clf = train(sents, labels)

  print 'number of features: ' + str(len(ft_xtractor.get_feature_names()))
  joblib.dump(ft_xtractor, 'ft_xtractor_moretrainSamples.pkl')
  joblib.dump(clf, 'clf_linear_all_words_ordered_prob.pkl')

def load_and_validate(ft_ext_file, clf_file):

  ft_xtractor = joblib.load(ft_ext_file)
  clf = joblib.load(clf_file)

  # print 'getting test data...'
  #valid_sents, expected_labels = get_test_data()

  sents, labels, pairs, recipeLength = get_tsp_test_data()

  prevItr = 0;
  tspResultSet = []
  for i in xrange(len(recipeLength)):


    itr = recipeLength[i][1]
    test_sents = sents[prevItr: prevItr + itr-1]
    print 'Recipe No ' + str(i+1)
    print 'Testing set size ' + str(itr)
    print 'Number of nodes ' + str(recipeLength[i][0])

    if len(test_sents) <= 1:
      print 'Skipping Recipe No ' + str(i + 1)
      tspResultSet.append([])
      prevItr += itr
      continue
    weights, pred_labels = test(test_sents, ft_xtractor, clf)

    edge_weights = tsp.pick_edge_weights(weights, pred_labels, pairs[prevItr: prevItr + itr-1], recipeLength[i][0])
    print 'Ordering for Recipe No ' + str(i+1) + ' is '
    tspResultSet.append(test_tsp_solver(edge_weights))
    prevItr += itr

  import pickle

  with open('SOAllWordsLinear.pickle', 'w') as f:
    pickle.dump(tspResultSet, f)

  f.close()



def test_tsp_solver(distances):

  # input = tsp.prepare_tsp_solver_input(distances)
  output = tsp.tsp_dyn_solver(distances)

  print 'solution'
  pprint(output)
  return output

def main():
  #run_classifier()
  # train_and_save()
  load_and_validate('ft_xtractor_moretrainSamples.pkl', 'clf_linear_all_words_ordered_prob.pkl')
  # findEstimator()
  # test_tsp_solver([[0, 1, 100, 200], [100, 0, 1000, 1], [100, 1000, 0, 200], [100, 100, 2, 0]])
  pass


if __name__ == '__main__':
  import time
  start_time = time.time()
  main()
  print time.time() - start_time, "seconds"
  print '#############'
