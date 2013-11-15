__author__ = 'gt'

import random
import codecs

###Parameters
testP           = 0.9 #rand >= testP for test sample
negativeP       = 0.45 # rand >= negativeP for negative sample
negativePTest   = 0.50 # rand >= negativeP for negative sample
trainFile       = '/Users/gt/CookScrap/VR/moretrainSamples.txt' #train samples generated and then used
testFile        = '/Users/gt/CookScrap/VR/moretestSamples.txt'  # same for test samples
archiveLocation = '/Users/gt/CookScrap/VR/transitions-2-pretty-V'
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

def prepare_training_data():
  import commands
  from BeautifulSoup import BeautifulSoup

  trainSamples = codecs.open(trainFile, 'w', encoding)
  testSamples  = codecs.open(testFile , 'w', encoding)

  files        = commands.getoutput('find ' + archiveLocation +
                                    ' -type f ')

  limit = 1
  for htmlf in files.rstrip().split('\n'):
    if limit == stopLimit: break
    limit +=1
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

    prev = None
    for tr in trs:
      tds = tr.findAll('td')
      if(len(tds) == 0): continue  #th row
      if prev == None:
        prev = tds[1].text #second column has text
        continue

      curr = tds[1].text

      tP = round(random.random(),2)
      nP = round(random.random(),2)

      if tP >= testP:
        if nP >= negativePTest:
          sample = curr + blockSeparator \
                   + prev + labelSeparator + '-'
        else:
          sample = prev + blockSeparator \
                   + curr + labelSeparator + '+'

        testSamples.write(sample + '\n')
      else:
        if nP >= negativeP:
          sample = curr + blockSeparator \
                   + prev + labelSeparator + '-'
        else:
          sample = prev + blockSeparator \
                   + curr + labelSeparator + '+'

        trainSamples.write(sample + '\n')

      prev = curr

  testSamples.close()
  trainSamples.close()

def prepare_test_data():
  pass

def preprocess():
  prepare_training_data()
  prepare_test_data()

def test_data_extraction():
  pass

def main():
  test_data_extraction()


if __name__ == '__main__':
  main()
  print '#############'

