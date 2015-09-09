import util
import operator
import re
import codecs

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
    if re.match(pattern, word):
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


def preprocessEnglishCorpus(CorpusFile, preprocessedFile):
    """
    func: preprocess corpus
    params: CorpusFile: corpus file path
    params: preprocessedFile: the output preprocessed file
    return: n/a
    """
    print 'preprocessing...'
    reader = open(CorpusFile, 'r')
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
            #for word in words:

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


def loadBilingualDictionary(bilingual_dict_file):
    with codecs.open(bilingual_dict_file, 'r', 'utf-8') as reader:
        entrylines = reader.readlines()
    ch2eng = {}
    eng2ch = {}
    count = 0
    for line in entrylines:
        count += 1
        print count
        chword = line.split('\t')[0]
        engtrans = line.split('\t')[-1].strip().strip('/').split('/')
        engwords = []
        # delete () description
        for engtran in engtrans:
            print engtran
            print line
            if not(engtran[0] == '(' and engtran[-1]==')'):
                engwords.append(engtran)
        ch2eng[chword] = engwords

    # building english to chinese dictionary
    for chword, engwords in ch2eng.iteritems():



if __name__ == '__main__':
    loadBilingualDictionary('../data/bilingual_dict.lst')
