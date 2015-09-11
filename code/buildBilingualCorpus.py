__author__ = 'yandixia'

import re
import codecs
import numpy as np


chinese_punctuation = u'\uFF01\uFF1F\uFF61\u3002\uFF02\uFF03\uFF04\uFF05\uFF06\uFF07\uFF08\uFF09\uFF0A\uFF0B\uFF0C\uFF0D\uFF0F\uFF1A\uFF1B\uFF1C\uFF1D\uFF1E\uFF20\uFF3B\uFF3C\uFF3D\uFF3E\uFF3F\uFF40\uFF5B\uFF5C\uFF5D\uFF5E\uFF5F\uFF60\uFF62\uFF63\uFF64\u3000\u3001\u3003\u3008\u3009\u300A\u300B\u300C\u300D\u300E\u300F\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301A\u301B\u301C\u301D\u301E\u301F\u3030\u303E\u303F\u2013\u2014\u2018\u2019\u201B\u201C\u201D\u201E\u201F\u2026\u2027\uFE4F\uFE51\uFE54\u00B7'


def chinese_word_filter(word):
    """
    func: preprocess word level
    params: word: the word to be preprocessed
    return: a list of seperated words. [] if word is to be removed.
    """
    # punctuations like [,.-'"/;]
    if word in chinese_punctuation:
        return []

    # filter for url
    isURL = word[:3] == 'www' or '.com' in word or '.org' in word \
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

    return [word]


def english_word_filter(word):
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
    if word in "\"'?/[.,-\/#!$%\^&\*;:{}=\-_`~()]/":
        return []

    # filter for url
    isURL = word[:3] == 'www' or '.com' in word or '.org' in word \
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

    return [word]


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
    open(preprocessedFile, 'w').close()
    while True:
        outputbuffer = []
        lines = reader.readlines(buffsize)
        if not lines:
            break
        else:
            buffcount += 1
            print 'building with ' + str(buffcount) + ' buffer.....'
        for line in lines:
            words = line.split()
            newwords = []
            for word in words:
                newwords.extend(english_word_filter(word))
            outputbuffer.append(' '.join(newwords) + '\n')
        print 'writing buffer...'
        with open(preprocessedFile, 'a') as writer:
            writer.writelines(outputbuffer)
    reader.close()


def preprocessChineseCorpus(CorpusFile, preprocessedFile):
    """
    func: preprocess corpus
    params: CorpusFile: corpus file path
    params: preprocessedFile: the output preprocessed file
    return: n/a
    """
    print 'preprocessing...'
    print 'reading corpus...'
    with codecs.open(CorpusFile, 'r', 'utf-8') as reader:
        corpuslines = reader.readlines()
    print 'reading corpus done...'

    buffsize = 1000000
    buffnum = len(corpuslines) / buffsize + 1
    codecs.open(preprocessedFile, 'w', 'utf-8').close()
    for i in xrange(buffnum):
        print 'building with ' + str(i) + ' buffer.....'
        outputbuffer = []
        bufflines = corpuslines[i * buffsize:(i + 1) * buffsize]
        for line in bufflines:
            words = line.split()
            newwords = []
            for word in words:
                newwords.extend(chinese_word_filter(word))
            outputbuffer.append(' '.join(newwords) + '\n')
        print 'writing buffer...'
        with codecs.open(preprocessedFile, 'a', 'utf-8') as writer:
            writer.writelines(outputbuffer)
    reader.close()

if __name__ == '__main__':
    preprocessChineseCorpus('../data/gigacorpus', '../data/pre_gigacorpus')