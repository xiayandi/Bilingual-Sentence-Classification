__author__ = 'yandixia'

import operator
import numpy as np
import cPickle
import codecs
import sys


def outputAllVocabList(filelist, encodings, vocabFile):
    """
    :func extracting the vocab in list of files and output the vocab to vocabFile
    :param filelist: a list of file paths
    :param encodings: the corresponding encodings for each file
    :param vocabFile: the output vocab file
    :return: n/a
    """
    vocab = []
    for i, foo in enumerate(filelist):
        with codecs.open(foo, 'r', encoding=encodings[i]) as reader:
            lines = reader.readlines()
        for line in lines:
            vocab.extend(line.split('\t')[1].split())
    sortedcheatvocab = sorted(set(vocab))
    with codecs.open(vocabFile, 'w', 'utf-8') as writer:
        writer.write('PADDING' + '\n')  # add padding into vocab
        for word in sortedcheatvocab:
            writer.write(word + '\n')


def construct_w2v(emb_dim, vocablist, w2vfile, w2vout):
    """
    func: construct the word embedding dictionary and store in numpy array
    :param emb_dim: dimension of word embeddings
    :param vocablist: vocab file
    :param w2vfile: original word embedding file
    :param w2vout: the output of the trimmed file
    :return:
    """
    print 'constructing w2v dictionary....'
    with open(w2vfile, 'r') as reader:
        w2vlines = reader.readlines()
    with open(vocablist, 'r') as reader:
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
    outcount = 0
    incount = 0
    allcount = len(w2vlines)
    vocabcount = len(w2idx)
    w2v = np.zeros((len(w2idx), emb_dim), dtype=np.float32)
    for i in xrange(2, len(w2vlines)):
        line = w2vlines[i]
        wd = line.split()[0]
        if wd in w2idx:
            vector = [np.float32(digit) for digit in line.split()[1:]]
            w2v[w2idx[wd]] = vector
            incount += 1
        else:
            outcount += 1
    print 'out of vocabulary count: ' + str(outcount)
    print 'in vocabulary count: ' + str(incount)
    print 'w2v count: ' + str(allcount)
    print 'vocab count: ' + str(vocabcount)

    print 'constructing w2v done!'

    w2v_output = open(w2vout, 'wb')
    cPickle.dump(w2v, w2v_output, -1)
    w2v_output.close()


def findUnkownWords(vocablist, w2vfile):
    """
    func: for statistic use, find all the unk words.
    :param vocablist:
    :param w2vfile:
    :return:
    """
    with open(w2vfile, 'r') as reader:
        w2vlines = reader.readlines()
    with open(vocablist, 'r') as reader:
        allvocablines = reader.readlines()
    vocablst = []
    for i in xrange(len(allvocablines)):
        wd = allvocablines[i].split()[0]
        vocablst.append(wd)
    w2vwordset = set()
    for i in xrange(2, len(w2vlines)):
        line = w2vlines[i]
        wd = line.split()[0]
        w2vwordset.add(wd)
    unkcount = 0
    for i, wd in enumerate(vocablst):
        if wd not in w2vwordset:
            unkcount += 1
            print wd
    print 'unk count: ' + str(unkcount)
    print 'all count: ' + str(len(vocablst))


def rundown(allw2v, files, encodings):
    trimmedw2v = '../exp/blg250.pkl'
    filelist = files
    assert len(files) == len(encodings)
    vocabFile = '../exp/vocab_bi.lst'
    outputAllVocabList(filelist, encodings, vocabFile)
    construct_w2v(250, vocabFile, allw2v, trimmedw2v)


def rundown_config(config_):
    trimmedw2v = '../exp/blg250.pkl'
    filelist = [config_.trainfile, config_.testfile, config_.validfile]
    allw2v = config_.allw2v
    encodings = ['utf8', 'utf8', 'utf8']
    vocabFile = '../exp/vocab_bi.lst'
    outputAllVocabList(filelist, encodings, vocabFile)
    construct_w2v(250, vocabFile, allw2v, trimmedw2v)


if __name__ == '__main__':
    allw2v = '../data/blg250.txt'  # bilingual embedding
    #allw2v = '../data/blg250_all_phrase.txt' # phrase based bilingual embedding
    #allw2v = '../data/ch_250.txt' # chinese embedding
    #files = ['../data/QC/translate/final_moses_eng2ch_train.seg', '../data/QC/Chinese_qc/finaltest.seg',
    #         '../data/QC/Chinese_qc/validset.seg']  # TREC translated data
    # files = ['../data/QC/TREC/trimengqctrain.phr', '../data/QC/Chinese_qc/finaltest.seg'] # TREC  tranlated data
    #files = ['../data/Semantic/movieReview/imdb/eng_train', '../data/Semantic/movieReview/Douban/test.dat.seg',
    #         '../data/Semantic/movieReview/Douban/validset.seg'] # MR data
    #files = ['../data/Semantic/movieReview/trans_imdb/moses_trans_mr_eng2ch_train.seg',
    #         '../data/Semantic/movieReview/Douban/test.dat.seg',
    #         '../data/Semantic/movieReview/Douban/validset.seg'] # MR translated data
    #files = ['../data/Semantic/productReview/train.dat', '../data/Semantic/productReview/test.dat.seg'] # PR data
    #files = ['../data/Event/English/train.dat', '../data/Event/Chinese/sub_test.dat.seg'] # event data
    #encodings = ['utf-8', 'utf-8', 'utf-8']
    #rundown(allw2v, files, encodings)









