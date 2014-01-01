__author__ = 'gt'
from sklearn.feature_extraction.text import  CountVectorizer

from nltk import word_tokenize
from nltk import pos_tag
from nltk import PorterStemmer
import re
import numpy as np
from scipy.sparse import csc_matrix, hstack, vstack, csr_matrix

from pprint import pprint
stemmer = PorterStemmer()

from gt.sbu.so.data import blockSeparator,transitionSeparator, imageSeparator
from gt.sbu.so.data import encoding, loadMatlabFile, matFile


punc_rx = re.compile(r'[^#A-Za-z0-9]+', re.DOTALL)
#POSTags looked up from http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
###Parameters
#Both stanford pos tagger and default nlkt pos tagger are using same labels for pos taggin
filterList = ['VBZ', 'VBP', 'VBN', 'VBG', 'VBD', 'VB', #Verb forms
                 # 'RB', 'RBR', 'RBS', 'WRB', #ADV
				# 'PRP', 'PRP$', 'WP', 'WP$',	#pronouns
        'DT',#Articles
              ] #Pos tags to be filtered
###

def nltk_filter(sent):
  b1, b2 = sent.split(blockSeparator)

  b1 = b1.rstrip()
  b1, t1 = b1.split(transitionSeparator)
  b2 = b2.rstrip()
  b2, t2 = b2.split(transitionSeparator)

  b1            = b1.lower()
  tokens        = word_tokenize(b1)
  filtered_sent = ' '
  for token in tokens:
    filtered_sent += '1'+ stemmer.stem(token) + ' '
  # for pos_t in pos_tags:
  #   if pos_t[1] in filterList:
  #     #filtered_sent += stemmer.stem(pos_t[0]) + ' '
  #     filtered_sent += '1' + stemmer.stem(pos_t[0]) + ' '

#note: 1 concat stemmer(word) == stemmer(1 concat word)

  b2            = b2.lower()
  tokens        = word_tokenize(b2)


  # for pos_t in pos_tags:
  #   if pos_t[1] in filterList:
  #     #filtered_sent += stemmer.stem(pos_t[0]) + ' '
  #     filtered_sent += '2' + stemmer.stem(pos_t[0]) + ' '

  for token in tokens:
    filtered_sent += '2' + stemmer.stem(token) + ' '

  return filtered_sent

def stanford_corenlp_filter(sent):
  from nltk.tag.stanford import POSTagger
  posTagger = POSTagger('/Users/gt/Downloads/'
                        'stanford-postagger-2013-06-20/models/'
                        'wsj-0-18-bidirectional-nodistsim.tagger',
                        '/Users/gt/Downloads/stanford-postagger-2013-06-20'
                        '/stanford-postagger-3.2.0.jar',encoding=encoding)

  b1, b2 = sent.split(blockSeparator)
  b2 = b2.rstrip()

  b1 = b1.lower()
  tokens = word_tokenize(b1)
  pos_tags = posTagger.tag(tokens)
  filtered_sent = ' '
  for pos_t in pos_tags:
    if pos_t[1] in filterList:
      # filtered_sent += stemmer.stem(pos_t[0]) + ' '
      filtered_sent += '1' + stemmer.stem(pos_t[0]) + ' '

      #note: 1 concat stemmer(word) == stemmer(1 concat word)

  b2 = b2.lower()
  tokens = word_tokenize(b2)
  pos_tags = posTagger.tag(tokens)

  for pos_t in pos_tags:
    if pos_t[1] in filterList:
      # filtered_sent += stemmer.stem(pos_t[0]) + ' '
      filtered_sent += '2' + stemmer.stem(pos_t[0]) + ' '

  return filtered_sent


def filter_text(sent):
  # return stanford_corenlp_filter(sent)
  sent = re.sub(punc_rx, ' ', sent)
  return nltk_filter(sent)
  # sents = sent.split(blockSeparator)
  # sent = sents[0] + ' ' + sents[1]
  # return sent

def get_entity_features(grid1, grid2):
  ret =[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

  transition_vocabulary = ['s', 'o', 'x', '-']
  for i in range(len(transition_vocabulary)):
    for j in range(len(transition_vocabulary)):
      for k in range(len(grid1)):
        if grid1[k] == transition_vocabulary[i] and \
            grid2[k] == transition_vocabulary[j]:
          ret[(i * 4) + (j)] += 1

  #Normalizing for different number of ingredients per recipe
  ret = [ x/float(len(grid1)) for x in ret]

  return ret

def get_features(sents, vec=1):
  if vec == 1:
    vec = CountVectorizer(min_df=1, binary=True, tokenizer=word_tokenize,
                        preprocessor=filter_text, ngram_range=(1,2) )
    # vec = TfidfVectorizer(min_df=1, tokenizer=word_tokenize,
    #                       preprocessor=filter_text, ngram_range=(1,2) )
    X   = vec.fit_transform(sents)
  else:
    X   = vec.transform(sents)

  Xentity = []
  ftMatrix = loadMatlabFile(matFile)
  XHog1 = csr_matrix([])
  XHog2 = csr_matrix([])

  for sent in sents:
    b1, b2 = sent.split(blockSeparator)
    b1 = b1.rstrip()
    b1, t1 = b1.split(transitionSeparator)
    t1, hogIdx1 = t1.split(imageSeparator)
    hogIdx1 = hogIdx1.rstrip()
    hogIdx1 = int(hogIdx1)
    b2 = b2.rstrip()
    b2, t2 = b2.split(transitionSeparator)
    t2, hogIdx2 = t2.split(imageSeparator)
    hogIdx2 = hogIdx2.rstrip()
    hogIdx2 = int(hogIdx2)

    Xentity.append(get_entity_features(t1.split(','), t2.split(',')))
    if XHog1.shape == (1,0):
      XHog1 = csr_matrix(ftMatrix[hogIdx1])
    else:
      XHog1 = vstack([XHog1, ftMatrix[hogIdx1]])

    if XHog2.shape == (1,0):
      XHog2 = csr_matrix(ftMatrix[hogIdx2])
    else:
      XHog2 = vstack([XHog2, ftMatrix[hogIdx2]])

  X = hstack([X, csc_matrix(Xentity)])
  X = hstack([X, csc_matrix(XHog1)])
  X = hstack([X, csc_matrix(XHog2)])

  #pprint(str(X))
  return vec, X

def test_features():
  sents = [
    'In a medium saucepan over medium-low heat, melt butter in buttermilk. Whisk in flour and spices, and cook until thickened, about 5 minutes. Turn off heat and add the cheeses, stirring until melted.#TRANSITION#o,-,x,-,-,-,-,-,-,-,-,-,x,-,x,-#IMAGE#3711#BLOCK#In a small skillet, heavily salt diced pork belly, and cook until crisp. Set on paper towels to drain and set aside. Get pasta boiling (sorry, no pics of pasta...), under cook by 1 minute. If pasta is done before sauce, toss with a little oil to prevent sticking.#TRANSITION#x,-,-,-,-,-,-,-,x,-,-,-,-,-,x,-#IMAGE#3710',
    '    In a small skillet, heavily salt diced pork belly, and cook until crisp. Set on paper towels to drain and set aside. Get pasta boiling (sorry, no pics of pasta...), under cook by 1 minute. If pasta is done before sauce, toss with a little oil to prevent sticking.#TRANSITION#x,-,-,-,-,-,-,-,x,-,-,-,-,-,x,-#IMAGE#3710#BLOCK#dd pasta to sauce, and add in tomatoes and diced pork belly. Put into greased 8x8 or 9x9 pan, and top with panko. Bake at 400F for 15 minutes, or until golden brown on top (broil if necessary).#TRANSITION#-,-,x,-,x,-,-,x,x,-,-,-,-,-,-,-#IMAGE#3712',
  ]
  vec, X = get_features(sents)

  print vec.get_params()
  print vec.get_feature_names()
  pprint(X)

def main():
  test_features()

if __name__ == '__main__':
  main()
  print '#############'