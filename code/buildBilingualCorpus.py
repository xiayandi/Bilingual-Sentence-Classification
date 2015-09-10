__author__ = 'yandixia'

import re


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
    if word in [',', '.', '-', '--', '\'', '\"', ';', '/']:
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
            print 'building with ' + str(buffcount) + ' buffer.....'
        for line in lines:
            words = line.split()
            for word in words:
                word_filter(word)

    reader.close()

if __name__ == '__main__':
    preprocessEnglishCorpus('../data/', '../exp/')