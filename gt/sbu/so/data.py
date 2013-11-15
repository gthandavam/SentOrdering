__author__ = 'gt'

import random
import codecs

###Parameters
testFileList    = '/Users/gt/CookScrap/VR/Experiments/testFileList'
trainFileList   = '/Users/gt/CookScrap/VR/Experiments/trainFileList'
valFileList     = '/Users/gt/CookScrap/VR/Experiments/valFileList'
negP            = 0.50 # rand >= negativeP for negative sample
trainFile       = '/Users/gt/CookScrap/VR/Experiments/moretrainSamples.txt' #train samples generated and then used
testFile        = '/Users/gt/CookScrap/VR/Experiments/moretestSamples.txt'  # same for test samples
validFile       = '/Users/gt/CookScrap/VR/Experiments/moreValidSamples.txt'
encoding        = 'utf-8' #encoding per website
stopLimit       = 612 #dev parameter - to control the generation process
blockSeparator  = ' #BLOCK# ' #separator for 2 blocks
labelSeparator  = ' #LABEL# ' # separates the block and the label
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

def main():
  extract_data()

if __name__ == '__main__':
  main()
  print '#############'

