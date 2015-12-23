__author__ = 'yandixia'

import sys
import word2vec
import process_qc
import cnn_model
import preprocess_qc


class config:
    def __init__(self, trainbase, trainlang, testbase, validbase, lexicon, phrase, cdep, logprefix):
        self.lexicon = lexicon
        self.phrase = phrase
        self.cdep = cdep
        self.logprefix = logprefix

        self.trainlang = trainlang
        if trainlang == 'eng':
            if phrase:
                self.allw2v = '../data/blg250_all_phrase.txt'
                self.trainbase = trainbase
                self.trainfile = trainbase + '.phr'
                if cdep:
                    self.traindep = self.trainfile + '.cdep'
                else:
                    self.traindep = self.trainfile + '.dep'
            else:
                self.allw2v = '../data/blg250.txt'
                self.trainbase = trainbase
                self.trainfile = trainbase
                if cdep:
                    self.traindep = self.trainfile + '.cdep'
                else:
                    self.traindep = self.trainfile + '.dep'
        elif trainlang == 'ch':
            self.allw2v = '../data/ch_250.txt'
            self.trainbase = trainbase
            self.trainfile = trainbase + '.seg'
            if cdep:
                self.traindep = self.trainbase + '.cdep'
            else:
                self.traindep = self.trainbase + '.dep'
        else:
            print 'wrong language!'
            sys.exit()

        self.testbase = testbase
        self.testfile = testbase + '.seg'
        if cdep:
            self.testdep = testbase + '.cdep'
        else:
            self.testdep = testbase + '.dep'

        self.validbase = validbase
        self.validfile = validbase + '.seg'
        if cdep:
            self.validdep = validbase + '.cdep'
        else:
            self.validdep = validbase + '.dep'


def script():
    trainbase = '../data/Event/English/sub_train.dat'
    translate_base = '../data/Event/translate/moses_train.dat'
    chtrainbase = '../data/Event/Chinese/rest_test.dat'
    testbase = '../data/Event/Chinese/sub_test.dat'
    validbase = '../data/Event/Chinese/validset'

    # Event: dep, lex
    cdep = False
    phr = False
    lex = False
    logprefix = '../exp/trans_dep'
    filter_hss = [[3], [3, 4], [3, 4, 5]]
    config_ = config(translate_base, 'ch', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_)
    cnn_model.script(config_, filter_hss)


def qc_script():
    trainbase = '../data/QC/TREC/trimengqctrain'
    translate_base = '../data/QC/translate/final_moses_eng2ch_train'
    testbase = '../data/QC/Chinese_qc/finaltest'
    validbase = '../data/QC/Chinese_qc/validset'

    # Event: dep
    cdep = False
    phr = False
    lex = False
    logprefix = '../exp/dep_dep'
    filter_hss = [[3], [3, 4], [3, 4, 5]]
    config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_)
    cnn_model.script(config_, filter_hss)


if __name__ == '__main__':
    qc_script()
