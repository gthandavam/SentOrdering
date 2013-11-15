__author__ = 'gt'
import commands
import codecs
from BeautifulSoup import BeautifulSoup
import re
from pprint import pprint
import random

import math

def nCr(n,r):
  if n < r:
    return 0
  f = math.factorial
  return f(n) / f(r) / f(n-r)

validP    = 0.80
testComb  = 0
trainComb = 0
validComb = 0
outliers  = [
  '/Users/gt/CookScrap/VR/transitions-2-pretty-V/beef-tenderloin-with-whisky-and-mushroom-cream.html',
  '/Users/gt/CookScrap/VR/transitions-2-pretty-V/eggnog-and-quail-egg-shooters.html',
  '/Users/gt/CookScrap/VR/transitions-2-pretty-V/fruit-pudding.html',
  '/Users/gt/CookScrap/VR/transitions-2-pretty-V/halloween-chicken-fingers.html',

]

testEmptyComb = [
  '/Users/gt/CookScrap/VR/transitions-2-pretty-V/chicken-soup-with-goji-berries-and-red-dates.html'
]

testFiles = '/Users/gt/CookScrap/VR/Experiments/testFileList'
trainFiles = '/Users/gt/CookScrap/VR/Experiments/trainFileList'
valFiles = '/Users/gt/CookScrap/VR/Experiments/valFileList'
######
archiveLocation = '/Users/gt/CookScrap/VR/transitions-2-pretty-V'
encoding = 'utf-8'
######

ctr = 0
testF = open(testFiles,'w')
trainF = open(trainFiles,'w')
valF = open(valFiles,'w')
categories = {}
files        = commands.getoutput('find ' + archiveLocation +
                                    ' -type f ')

regex = re.compile('please be sure to view these other(\w+) recipes')

regex1 = re.compile('also, you will love these([\w\b\)\(]+) recipes')

for htmlf in files.rstrip().split('\n'):
  if htmlf in outliers:
    # print htmlf + ' outlier'
    continue
  f = codecs.open(htmlf, 'r', encoding)
  text = ''
  for line in f.readlines():
    line = line.replace('\n', ' ')
    text +=line

  f.close()
  soup = BeautifulSoup(text)
  tables = soup.findAll('table')

  trs = tables[0].findAll('tr')

  try:
    testLine  = trs[len(trs)-1].findAll('td')[1].text
  except BaseException:
    # print htmlf + ' outlier'
    continue



  testLine = testLine.lower()
  matches = regex.findall(testLine)
  matches1 = regex1.findall(testLine)



  if len(matches)== 0:
    if htmlf in testEmptyComb:
      continue
    testF.write(htmlf + '\n')
    print htmlf + ' test '
    testComb += nCr(len(trs)-1, 2)

    # if len(matches1) == 0:
    #   pass
    #   #testF.write(htmlf+'\n')
    # else:
    #   print htmlf

  else:

    vP = round(random.random(), 2)
    ctr +=1
    if categories.has_key(matches[0]):

      categories[matches[0]] += 1
    else:
      categories[matches[0]] = 1

    if vP > validP:
      valF.write(htmlf+'\n')
      print htmlf + ' validation'
      validComb += nCr(len(trs) - 1, 2)
    else:
      trainF.write(htmlf+'\n')
      print htmlf + ' train'
      trainComb += nCr(len(trs) - 1, 2)


print "test comb " + str(testComb)
print "train comb " + str(trainComb)
print "valid comb " + str(validComb)


"""
test comb 2484
train comb 26532
valid comb 5879
"""