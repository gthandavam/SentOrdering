__author__ = 'gt'

import random
import codecs
import numpy as np

###Parameters
testFileList    = '/Users/gt/CookScrap/VR/Experiments/testFileList'
trainFileList   = '/Users/gt/CookScrap/VR/Experiments/trainFileList'
valFileList     = '/Users/gt/CookScrap/VR/Experiments/valFileList'
negP            = 0.51 # rand >= negativeP for negative sample
trainFile       = '/Users/gt/CookScrap/VR/Experiments/moretrainSamples.txt' #train samples generated and then used
trainTSPFile    = '/Users/gt/CookScrap/VR/Experiments/moreTSPtrainSamplesCleanse1.txt'
# testFile        = '/Users/gt/CookScrap/VR/gt/ml/features/AdjtestSamples.txt'
testFile        = '/Users/gt/CookScrap/VR/Experiments/moretestSamples.txt'  # same for test samples
testTSPFile     = '/Users/gt/CookScrap/VR/Experiments/moreTSPtestSamplesCleanse1.txt'
validFile       = '/Users/gt/CookScrap/VR/Experiments/moreValidSamples.txt'
validTSPFile    = '/Users/gt/CookScrap/VR/Experiments/moreTSPValidSamplesCleanse1.txt'
encoding        = 'utf-8' #encoding per website
stopLimit       = 612 #dev parameter - to control the generation process
blockSeparator  = '#BLOCK#' #separator for 2 blocks
labelSeparator  = '#LABEL#' # separates the block and the label
pairSeparator   = '#PAIR#'
recipeSeparator = '#RECIPE#'
transitionSeparator = '#TRANSITION#'
###Parameters

def get_experiment_data(expFile):
  expF   = codecs.open(expFile, 'r',encoding)
  sents  = []
  labels = []

  for line in expF.readlines():

    sent, label = line.split(labelSeparator)
    label = label.rstrip()
    sents.append(sent)
    labels.append(label)

  expF.close()
  return sents, labels

def get_tsp_experiment_data(expFile):
  expF   = codecs.open(expFile, 'r',encoding)
  sents  = []
  labels = []
  pairs  = []
  recipeLength = []


  for line in expF.readlines():
    if line.find(recipeSeparator) != -1:
      line.rstrip()
      x,y = line.split(recipeSeparator)
      x,y = x.split(',')
      recipeLength.append((int(x),int(y)))
      continue


    sent, label = line.split(labelSeparator)
    label = label.rstrip()
    label, pair  = label.split(pairSeparator)
    pair  = pair.rstrip()
    sents.append(sent)
    labels.append(label)
    pairs.append(str(pair)) #comma separated


  expF.close()
  return sents, labels, pairs, recipeLength

def get_training_data():
  return get_experiment_data(trainFile)

def get_test_data():
  return get_experiment_data(testFile)

def get_validation_data():
  return get_experiment_data(validFile)

def prepare_experiment_data(inpFileList, outFile):

  from BeautifulSoup import BeautifulSoup

  samples      = codecs.open(outFile, 'w', encoding)
  files        = open(inpFileList)

  for htmlf in files.readlines():

    htmlf = htmlf.rstrip()
    htmlf = htmlf.replace('\n','')
    f = codecs.open(htmlf, 'r', encoding)
    text = ''
    for line in f.readlines():
      line = line.replace('\n', ' ')
      text += line
    f.close()
    soup = BeautifulSoup(text)
    tables = soup.findAll('table')
    #first table contains the steps
    trs = tables[0].findAll('tr')


    #ignoring the th row from the table
    for i in range(1,len(trs)):
      predecessor = trs[i].findAll('td')[1].text
      for j in range(i+1, len(trs)):
        successor = trs[j].findAll('td')[1].text
        tP = round(random.random(),2)
        if tP >= negP:
          sample = successor + blockSeparator \
                   + predecessor + labelSeparator + '-'
        else:
          sample = predecessor + blockSeparator \
                   + successor + labelSeparator + '+'

        samples.write(sample + '\n')

  files.close()
  samples.close()


def display_tsp_results(inpFileList, tspResultSet):
  from BeautifulSoup import BeautifulSoup

  files = open(inpFileList)
  itr = 0

  for htmlf in files.readlines():

    htmlf = htmlf.rstrip()
    htmlf = htmlf.replace('\n', '')
    f = codecs.open(htmlf, 'r', encoding)
    text = ''
    for line in f.readlines():
      line = line.replace('\n', ' ')
      text += line
    f.close()

    outhtmlf = htmlf.replace('transitions-2-pretty-V', 'Results/Exp1')
    outf = codecs.open(outhtmlf, 'w', encoding)

    name = outhtmlf.split('/')[-1]
    my_html = '<html>'
    my_html = '<head> <meta charset=\'' + encoding + '\'/>'
    my_html += ('<title>' + name + '</title> </head>')
    my_html += ('<body>')

    my_html += ('<table border ="1">')
    my_html += ('<tr>')
    my_html += ('<th> Gold Standard Order </th> <th> Experiment Order </th>')
    my_html += ('</tr>')

    soup = BeautifulSoup(text)
    tables = soup.findAll('table')
    #first table contains the steps
    trs = tables[0].findAll('tr')

    if len(trs) <= 1:
      print 'Skipping ' + htmlf

    print ' recipe from ' + htmlf
    i=1
    for j in tspResultSet[itr]:
      my_html += '<tr> <td> ' + trs[i].findAll('td')[1].text + ' </td> '
      my_html += ' <td> ' + trs[j + 1].findAll('td')[1].text + ' </td> </tr> '
      i += 1


    outf.write(my_html)
    outf.close()
    itr += 1

  files.close()


def repl(m):
  one = m.group(1)
  two = m.group(2)
  three = m.group(3)
  return one + two + ' ' + three

def prepare_tsp_experiment_data(inpFileList, outFile):

  from BeautifulSoup import BeautifulSoup
  import json,re

  samples      = codecs.open(outFile, 'w', encoding)
  files        = open(inpFileList)

  for htmlf in files.readlines():

    htmlf = htmlf.rstrip()
    htmlf = htmlf.replace('\n','')
    f = codecs.open(htmlf, 'r', encoding)

    jsonf = htmlf.replace('.html', '.json')
    jsonf = jsonf.replace('transitions-2', 'transitionsJson2')
    jf = open(jsonf, 'r')
    transitions = ''
    for line in jf.readlines():
      transitions += line
    jf.close()

    cTransitions = json.loads(transitions)

    text = ''
    for line in f.readlines():
      line = line.replace('\n', ' ')
      re.sub(r'([\w\W])(\.+)([\w\W])', repl, line)
      text += line
    f.close()

    soup = BeautifulSoup(text)
    tables = soup.findAll('table')
    #first table contains the steps
    trs = tables[0].findAll('tr')

    sampleCtr = 0
    #ignoring the th row from the table
    for i in range(1,len(trs)):
      predecessor = trs[i].findAll('td')[1].text
      for j in range(i+1, len(trs)):
        successor = trs[j].findAll('td')[1].text
        tP = round(random.random(),2)
        if tP >= negP:
          sample = successor + transitionSeparator + ','.join(cTransitions[j-1]) + blockSeparator \
                   + predecessor + transitionSeparator + ','.join(cTransitions[i-1]) + labelSeparator + '-' \
                   + pairSeparator + str(j) + ',' + str(i)
        else:
          sample = predecessor + transitionSeparator + ','.join(cTransitions[i-1]) + blockSeparator \
                   + successor + transitionSeparator + ','.join(cTransitions[j-1]) + labelSeparator + '+' \
                   + pairSeparator + str(i) + ',' + str(j)
        sampleCtr += 1
        samples.write(sample + '\n')

    samples.write( str(len(trs)-1) + ',' + str(sampleCtr) + recipeSeparator + '\n' )

  files.close()
  samples.close()


def get_tsp_test_data():
  return get_tsp_experiment_data(testTSPFile)

def get_tsp_train_data():
  return get_tsp_experiment_data(trainTSPFile)

def get_tsp_validation_data():
  return get_tsp_experiment_data(validTSPFile)

def prepare_validation_data():
  prepare_experiment_data(valFileList, validFile)

def prepare_train_data():
  prepare_experiment_data(trainFileList, trainFile)

def prepare_test_data():
  prepare_experiment_data(testFileList, testFile)

def extract_data():
  prepare_train_data()
  prepare_validation_data()
  prepare_test_data()

def test_data_extraction():
  prepare_test_data()

def get_stat(expFile):
  f = codecs.open(expFile, 'r', encoding)
  p = 0
  ctr = 0
  for line in f.readlines():
    ctr += 1


    label = line.split(labelSeparator)[1]
    label = label.rstrip()
    # print label
    if label == '+':
      p += 1

  print "Number of positives " + str(p)
  print "Number of samples " + str(ctr)


def main():
  # extract_data()
  # get_stat(trainFile)
  # get_stat(validFile)
  # get_stat(testFile)
  prepare_tsp_experiment_data(trainFileList, trainTSPFile)


if __name__ == '__main__':
  main()
  print '#############'

'''
Number of positives 13050
Number of samples 26468
Number of positives 2865
Number of samples 5943
Number of positives 1280
Number of samples 2484
#############
'''