__author__ = 'yandixia'

import operator
import numpy as np
import cPickle
import codecs
import sys


def outputAllVocabList():
    trainFile = '../data/QC/Chinese_qc/finaltrain'
    testFile = '../data/QC/Chinese_qc/finaltest'
    cheatvocab = []
    with codecs.open(trainFile, 'r', 'utf-8') as reader:
        lines = reader.readlines()
    for line in lines:
        cheatvocab.extend(line.split('\t')[1].split())
    with codecs.open(testFile, 'r', 'utf-8') as reader:
        lines = reader.readlines()
    for line in lines:
        cheatvocab.extend(line.split('\t')[1].split())
    sortedcheatvocab = sorted(set(cheatvocab))
    with codecs.open('../exp/vocab_ch_qc.lst', 'w', 'utf-8') as writer:
        writer.write('PADDING' + '\n')  # add padding into vocab
        for word in sortedcheatvocab:
            writer.write(word + '\n')


def construct_w2v(emb_dim, w2vfile, w2vout):
    # construct the word embedding dictionary
    # and store in numpy array
    print 'constructing w2v dictionary....'
    with codecs.open(w2vfile, 'r') as reader:
        w2vlines = reader.readlines()
    with codecs.open('../exp/vocab_ch_qc.lst', 'r') as reader:
        allvocablines = reader.readlines()

    # create word to all vocab index dictionary
    print 'creating all word to index dictionary...'
    w2idx = {}
    vocablst = []
    for i in xrange(len(allvocablines)):
        wd = allvocablines[i].split()[0]
        w2idx[wd] = i
        vocablst.append(wd)

    # create w2v matrix
    print 'creating word to vector matrix...'
    w2v = np.zeros((len(w2idx), emb_dim), dtype=np.float32)
    for i in xrange(2, len(w2vlines)):
        line = w2vlines[i]
        wd = line.split()[0]
        if wd in w2idx:
            vector = [np.float32(digit) for digit in line.split()[1:]]
            w2v[w2idx[wd]] = vector

    print 'constructing w2v done!'

    w2v_output = open(w2vout, 'wb')
    cPickle.dump(w2v, w2v_output, -1)
    w2v_output.close()


def rundown():
    rawlist = [
        '../data/w2v_250.txt',
    ]
    processedlist = [
        '../exp/ch_250.pkl',
    ]
    for rf, pf in zip(rawlist, processedlist):
        outputAllVocabList()
        construct_w2v(250, rf, pf)


if __name__ == '__main__':
    rundown()









