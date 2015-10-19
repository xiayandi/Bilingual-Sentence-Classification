__author__ = 'yandixia'

import util
import operator
import codecs

"""
This module provide a way to build bilingual (English and Chinese) dictionary
"""


def _buildEnglishVocab(segCorpusFile, vocabFile):
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
            print 'building with ' + str(buffcount) + ' buffer.....'
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
            voc_line = word + '\t' + str(count) + '\n'
            writer.write(voc_line)
    print 'output vocabulary done.'


def trimBilingualDictionary(bilingual_dict_file, trimed_bilingual_dict_file):
    with codecs.open(bilingual_dict_file, 'r', 'utf-8') as reader:
        entrylines = reader.readlines()
    newlines = []
    for line in entrylines:
        chword = line.split('\t')[0]
        enpart = line.split('\t')[1].strip().strip('/')
        enwords = enpart.split('/')
        trimedenwords = []
        for ewd in enwords:
            if '(' in ewd or len(ewd.split()) > 1:
                continue
            else:
                trimedenwords.append(ewd)
        if len(trimedenwords) != 0:
            newlines.append(chword + '\t/' + '/'.join(trimedenwords) + '/\n')
    with codecs.open(trimed_bilingual_dict_file, 'w', 'utf-8') as writer:
        writer.writelines(newlines)



def loadBilingualDictionary(bilingual_dict_file):
    """
    :func: loading bilingual dictionary from LDC Chinese-English dictionary
    :param bilingual_dict_file: LDC Chinese-English dictionary
    :return: chinese to english dictionary and english to chinese dictionary.
            the structure is a dictionary with list entry
    """
    with codecs.open(bilingual_dict_file, 'r', 'utf-8') as reader:
        entrylines = reader.readlines()
    ch2eng = {}
    eng2ch = {}
    for line in entrylines:
        chword = line.split('\t')[0]
        engtrans = line.split('\t')[-1].strip().strip('/').split('/')
        engwords = []
        # delete () description
        for engtran in engtrans:
            if not (engtran[0] == '(' and engtran[-1] == ')'):
                engwords.append(engtran)
        ch2eng[chword] = engwords

    # building english to chinese dictionary
    for chword, engwords in ch2eng.iteritems():
        for engword in engwords:
            # uengword = engword.decode('utf-8')
            #engword = uengword.encode('ascii', 'ignore')
            if engword not in eng2ch:
                eng2ch[engword] = set()
                eng2ch[engword].add(chword)
            else:
                eng2ch[engword].add(chword)

    # convert set in each entry into list
    for engword, chset in eng2ch.iteritems():
        eng2ch[engword] = list(chset)

    return eng2ch, ch2eng


if __name__ == '__main__':
    trimBilingualDictionary('../data/bilingual_dict.lst', '../data/trimed_bilingual_dict.lst')
