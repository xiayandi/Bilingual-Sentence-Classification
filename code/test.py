"""
this module is for test only
"""
import codecs

with codecs.open('../exp/testdata', 'r', 'utf-8') as reader:
    datalines = reader.readlines()
for line in datalines:
    print line
