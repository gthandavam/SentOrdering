__author__ = 'gt'
from sklearn.svm import SVC
from sklearn.externals import joblib
from gt.sbu.so.ft_extraction import get_features

def save_model(f_ext, classifier):
  joblib.dump(f_ext, 'test_ft_ext.pkl')
  joblib.dump(classifier, 'test_classifier.pkl')

def test_load():
  f_ext = joblib.load('test_ft_ext.pkl')
  classifier = joblib.load('test_classifier.pkl')
  test1 = ['second goyyale document.     #BLOCK# Testing',]

  X = f_ext.transform(test1)
  print f_ext.get_feature_names()
  print X.todense()
  print classifier.predict(X)


def main():
  corpus = [
    'This is the first document. #BLOCK# Second sentence looks interesting',
    'This is the second second document.     #BLOCK# Testing',
    'And the third one.     #BLOCK# Does it work',
    'Is this the first document ?     #BLOCK# Yes it does',
  ]
  labels = ['+', '+', '-', '-']
  vec, X = get_features(corpus)
  clf = SVC(kernel='rbf')
  clf.fit(X, labels)

  testSample = ['And the third one.     #BLOCK# Does it work']
  Xtest = vec.transform(testSample)
  print Xtest.todense()
  print clf.predict(Xtest)
  save_model(vec, clf)


if __name__ == '__main__':
  # main()
  test_load()

