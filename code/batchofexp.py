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
                self.trainfile = trainbase+'.phr'
                if cdep:
                    self.traindep = self.trainfile+'.cdep'
                else:
                    self.traindep = self.trainfile+'.dep'
            else:
                self.allw2v = '../data/blg250.txt'
                self.trainbase = trainbase
                self.trainfile = trainbase
                if cdep:
                    self.traindep = self.trainfile+'.cdep'
                else:
                    self.traindep = self.trainfile+'.dep'
        elif trainlang == 'ch':
                self.allw2v = '../data/ch_250.txt'
                self.trainbase = trainbase
                self.trainfile = trainbase+'.seg'
                if cdep:
                    self.traindep = self.trainbase+'.cdep'
                else:
                    self.traindep = self.trainbase+'.dep'
        else:
            print 'wrong language!'
            sys.exit()

        self.testbase = testbase
        self.testfile = testbase+'.seg'
        if cdep:
            self.testdep = testbase+'.cdep'
        else:
            self.testdep = testbase+'.dep'

        self.validbase = validbase
        self.validfile = validbase+'.seg'
        if cdep:
            self.validdep = validbase+'.cdep'
        else:
            self.validdep = validbase+'.dep'


def script():
    trainbase = '../data/Semantic/productReview/train.dat'
    translate_base = '../data/Semantic/productReview/moses_train.dat'
    testbase = '../data/Semantic/productReview/test.dat'
    validbase = '../data/Semantic/productReview/validset'

    preprocess_qc.splitValidOut(testbase, 314)

    # PR dep, phr, lex
    cdep = False
    phr = True
    lex = True
    logprefix = 'dep_phr_lex'
    filter_hss = [[1,3],[1,3,4],[1,3,4,5]]
    config_2 = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_2)
    process_qc.datasetConstructRundown_config(config_2)
    cnn_model.script(config_2, filter_hss)

    # PR dep, lex
    cdep = False
    phr = False
    lex = True
    logprefix = 'dep_lex'
    filter_hss = [[1,3],[1,3,4],[1,3,4,5]]
    config_3 = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_3)
    process_qc.datasetConstructRundown_config(config_3)
    cnn_model.script(config_3, filter_hss)

    # PR dep
    cdep = False
    phr = False
    lex = False
    logprefix = 'dep'
    filter_hss = [[3],[3,4],[3,4,5]]
    config_4 = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_4)
    process_qc.datasetConstructRundown_config(config_4)
    cnn_model.script(config_4, filter_hss)

    # PR translated
#   cdep = False
#   phr = False
#   lex = False
#   logprefix = 'translated'
#   filter_hss = [[3],[3,4],[3,4,5]]
#   config_5 = config(translate_base, 'ch', testbase, validbase, lex, phr, cdep, logprefix)
#   word2vec.rundown_config(config_5)
#   process_qc.datasetConstructRundown_config(config_5)
#   cnn_model.script(config_5, filter_hss)


if __name__ == '__main__':
    script()




