import util
import operator
import re

"""
This module provide a way to build bilingual (English and Chinese) dictionary
from www.iciba.com
"""

def word_filter(word):
    """
    func: preprocess word level
    params: word: the word to be preprocessed
    return: a list of seperated words. [] if word is to be removed.
    """
    # normal word
    pattern = "^[a-zA-z]+$"
    if re.match(pattern, word)
        return [word]

    # punctuations like [,.-'"/;]
    if word in [',','.','-','--','\'','\"',';','/']:
        return []

    # filter for url
    isURL = word[:3] == 'www' or '.com' in word or '.org' in word\
            or '.uk' in word
    if isURL:
        return ['URL']

    # filter for year
    pattern = '^[12][0-9]{3}$'
    if re.match(pattern, word):
        return ['YEAR']

    # filter for numbers
    pattern = "^[0-9\,\.\-]+$"
    if re.match(pattern, word):
        return ['NUMBER']

    # split word
    if '-' in word:
        words = []
        parts = word.split('-')
        for part in parts:
            if parts:
                words.extend(word_filter(parts))
        return words




def preprocessEnglishCorpus(CorpusFile, preprocessedFile):
    """
    func: preprocess corpus
    params: CorpusFile: corpus file path
    params: preprocessedFile: the output preprocessed file
    return: n/a
    """
    print 'preprocessing...'
    reader = open(segCorpusFile, 'r')
    buffsize = 250000000
    buffcount = 0
    outputbuffer = []
    while True:
        lines = reader.readlines(buffsize)
        if not lines:
            break
        else:
            buffcount += 1
            print 'building with '+str(buffcount)+' buffer.....'
        for line in lines:
            words = line.split()
            for word in words:

    reader.close()



def buildEnglishVocab(segCorpusFile, vocabFile):
    """
    func: get vocab from segmented English corpus file
    param: segCorpusFile: a segmented corpus file path. sentence per line.
    param: vocabFile: the output file of the vocab
    return: vocab dictionary
    """
    vocab = {}
    print 'building vocabulary...'
    reader = open(segCorpusFile, 'r')
    buffsize = 250000000
    buffcount = 0
    while True:
        lines = reader.readlines(buffsize)
        if not lines:
            break
        else:
            buffcount += 1
            print 'building with '+str(buffcount)+' buffer.....'
        for line in lines:
            words = line.split()
            for word in words:
                util.insertDict(word, vocab)
    reader.close()
    print 'building vocabulary done.'
    print 'output vocabulary...'
    sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
    with open(vocabFile, 'w') as writer:
        for word, count in sorted_vocab:
            voc_line = word+'\t'+str(count)+'\n'
            writer.write(voc_line)
    print 'output vocabulary done.'


if __name__ == '__main__':
    buildEnglishVocab('../data/1bwlmb', '../data/eng_vocab.lst')
