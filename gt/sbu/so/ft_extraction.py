__author__ = 'gt'
from sklearn.feature_extraction.text import  CountVectorizer

from nltk import word_tokenize
from nltk import pos_tag
from nltk import PorterStemmer
import re

from pprint import pprint
stemmer = PorterStemmer()

from gt.sbu.so.data import blockSeparator
from gt.sbu.so.data import encoding


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
  b2 = b2.rstrip()

  b1            = b1.lower()
  tokens        = word_tokenize(b1)
  pos_tags      = pos_tag(tokens)
  filtered_sent = ' '
  for token in tokens:
    filtered_sent += '1'+token + ' '
  # for pos_t in pos_tags:
  #   if pos_t[1] in filterList:
  #     #filtered_sent += stemmer.stem(pos_t[0]) + ' '
  #     filtered_sent += '1' + stemmer.stem(pos_t[0]) + ' '

#note: 1 concat stemmer(word) == stemmer(1 concat word)

  b2            = b2.lower()
  tokens        = word_tokenize(b2)
  pos_tags      = pos_tag(tokens)

  # for pos_t in pos_tags:
  #   if pos_t[1] in filterList:
  #     #filtered_sent += stemmer.stem(pos_t[0]) + ' '
  #     filtered_sent += '2' + stemmer.stem(pos_t[0]) + ' '

  for token in tokens:
    filtered_sent += '2' + token + ' '

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


def get_features(sents):
  vec = CountVectorizer(min_df=1, binary=True, tokenizer=word_tokenize,
                        preprocessor=filter_text, ngram_range=(1) )
  # vec = TfidfVectorizer(min_df=1, tokenizer=word_tokenize,
  #                       preprocessor=filter_text, ngram_range=(1,2) )
  X   = vec.fit_transform(sents)

  #pprint(str(X))
  return vec, X


def test_features():
  corpus = [
    'This is the first document. #BLOCK# Second sentence looks interesting',
    'This is the second second document.     #BLOCK# Testing',
    'And the third one.     #BLOCK# Does it work',
    'Is this the first document ?     #BLOCK# Yes it does',
  ]
  vec, X = get_features(corpus)



  print vec.get_params()
  print vec.get_feature_names()
  pprint(X.todense())

def main():
  test_features()


if __name__ == '__main__':
  main()
  print '#############'

