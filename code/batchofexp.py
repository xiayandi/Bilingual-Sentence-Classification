__author__ = 'yandixia'
"""
This module run experiments on different models and tasks.
"""

import sys
import word2vec
import process_qc
import cnn_model
import preprocess_qc


class config:
    def __init__(self, trainbase, trainlang, testbase, validbase, lexicon, phrase, cdep, logprefix, usefscore=False):
        self.lexicon = lexicon
        self.phrase = phrase
        self.cdep = cdep
        self.logprefix = logprefix
        self.usefscore = usefscore

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


def qc_script():
    trainbase = '../data/QC/TREC/trimengqctrain'
    translate_base = '../data/QC/translate/final_moses_eng2ch_train'
    testbase = '../data/QC/Chinese_qc/finaltest'
    validbase = '../data/QC/Chinese_qc/validset'
    chtrainbase = '../data/QC/Chinese_qc/finaltrain.new'

    # data quantity
    logprefix = '../exp/qc_data_quantity'
    accs = []
    open(logprefix, 'w').close()
    for i in xrange(1, 11):
        cdep = False
        dep = True
        phr = True
        lex = True
        config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
        word2vec.rundown_config(config_)
        process_qc.datasetConstructRundown_config(config_, dep, eng_portion=i)

        # general running
        batch_size = 120
        feature_map = 110
        filter_hs = [3]
        accs.append(cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon))
    with open(logprefix, 'w') as writer:
        for acc in accs:
            writer.write(str(acc) + '\t')

    """ running example
    cdep = False
    dep = True
    phr = True
    lex = True
    hasmlphidden = False
    logprefix = '../exp/qc_data_quantity'
    filter_hss = [[3], [3, 4], [3, 4, 5]]
    config_ = config(translate_base, 'ch', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_, dep)
    cnn_model.script(config_, filter_hss, hasmlphidden)

    # general running
    batch_size = 120
    feature_map = 110
    filter_hs = [3]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon)
    """


def qc_script2():
    trainbase = '../data/QC/TREC/trimengqctrain'
    translate_base = '../data/QC/translate/final_moses_eng2ch_train'
    testbase = '../data/QC/Chinese_qc/finaltest'
    validbase = '../data/QC/Chinese_qc/validset'
    chtrainbase = '../data/QC/Chinese_qc/finaltrain.new'

    cdep = False
    dep = True
    phr = True
    lex = True
    hasmlphidden = False
    logprefix = '../exp/qc_f_lex_phr'
    filter_hss = [[1, 3], [1, 3, 4], [1, 3, 4, 5]]
    config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_, dep)
    cnn_model.structure_script(config_, filter_hss, hasmlphidden)

    cdep = False
    dep = True
    phr = False
    lex = False
    hasmlphidden = False
    logprefix = '../exp/qc_f_trans_dcnn'
    filter_hss = [[3], [3, 4], [3, 4, 5]]
    config_ = config(translate_base, 'ch', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_, dep)
    cnn_model.structure_script(config_, filter_hss, hasmlphidden)

    cdep = False
    dep = False
    phr = False
    lex = False
    hasmlphidden = False
    logprefix = '../exp/qc_f_trans_cnn'
    filter_hss = [[3], [3, 4], [3, 4, 5]]
    config_ = config(translate_base, 'ch', testbase, validbase, lex, phr, cdep, logprefix)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_, dep)
    cnn_model.structure_script(config_, filter_hss, hasmlphidden)

    """
    # general running
    batch_size = 120
    feature_map = 110
    filter_hs = [3]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon)
    """


def mr_script():
    trainbase = '../data/Semantic/movieReview/imdb/eng_train'
    translate_base = '../data/Semantic/movieReview/trans_imdb/moses_trans_mr_eng2ch_train'
    testbase = '../data/Semantic/movieReview/Douban/test.dat.new'
    validbase = '../data/Semantic/movieReview/Douban/validset'

    # data quantity
    logprefix = '../exp/mr_data_quantity'
    accs = []
    open(logprefix, 'w').close()
    for i in xrange(1, 11):
        cdep = False
        dep = True
        phr = True
        lex = True
        config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
        word2vec.rundown_config(config_)
        process_qc.datasetConstructRundown_config(config_, dep, eng_portion=i)

        # general running
        batch_size = 240
        feature_map = 70
        filter_hs = [3]
        accs.append(cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon))

    with open(logprefix, 'w') as writer:
        for acc in accs:
            writer.write(str(acc) + '\t')
    """
    # general running
    batch_size = 240
    feature_map = 70
    filter_hs = [3]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon)
    """


def pr_script():
    trainbase = '../data/Semantic/productReview/train.dat'
    translate_base = '../data/Semantic/productReview/moses_train.dat'
    testbase = '../data/Semantic/productReview/test.dat.new'
    validbase = '../data/Semantic/productReview/validset'

    # data quantity
    logprefix = '../exp/pr_data_quantity'
    accs = []
    open(logprefix, 'w').close()
    for i in xrange(1, 11):
        cdep = False
        dep = True
        phr = True
        lex = True
        config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
        word2vec.rundown_config(config_)
        process_qc.datasetConstructRundown_config(config_, dep, eng_portion=i)

        # general running
        batch_size = 180
        feature_map = 50
        filter_hs = [3]
        accs.append(cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon))

    with open(logprefix, 'w') as writer:
        for acc in accs:
            writer.write(str(acc) + '\t')
    """
    # general running
    batch_size = 180
    feature_map = 50
    filter_hs = [3]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon)
    """


def event_script():
    trainbase = '../data/Event/English/sub_train.dat'
    translate_base = '../data/Event/translate/moses_train.dat'
    chtrainbase = '../data/Event/Chinese/rest_test.dat'
    testbase = '../data/Event/Chinese/sub_test.dat'
    validbase = '../data/Event/Chinese/validset'

    # data quantity
    logprefix = '../exp/event_data_quantity'
    accs = []
    open(logprefix, 'w').close()
    for i in xrange(1, 11):
        cdep = False
        dep = True
        phr = True
        lex = True
        config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
        word2vec.rundown_config(config_)
        process_qc.datasetConstructRundown_config(config_, dep, eng_portion=i)

        # general running
        batch_size = 160
        feature_map = 130
        filter_hs = [3]
        accs.append(cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon))

    with open(logprefix, 'w') as writer:
        for acc in accs:
            writer.write(str(acc) + '\t')
    """
    # general running
    batch_size = 160
    feature_map = 130
    filter_hs = [3]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon)
    """


def event_script_mono():
    trainbase = '../data/Event/English/sub_train.dat'
    translate_base = '../data/Event/translate/moses_train.dat'
    chtrainbase = '../data/Event/Chinese/rest_test.dat'
    testbase = '../data/Event/Chinese/sub_test.dat'
    validbase = '../data/Event/Chinese/validset'

    # data quantity
    logprefix = '../exp/event_data_quantity'
    accs = []
    open(logprefix, 'w').close()
    for i in xrange(1, 11):
        cdep = False
        dep = True
        phr = True
        lex = True
        config_ = config(trainbase, 'eng', testbase, validbase, lex, phr, cdep, logprefix)
        word2vec.rundown_config(config_)
        process_qc.datasetConstructRundown_config(config_, dep, eng_portion=i)

        # general running
        batch_size = 160
        feature_map = 130
        filter_hs = [3]
        accs.append(cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon))

    with open(logprefix, 'w') as writer:
        for acc in accs:
            writer.write(str(acc) + '\t')
    """
    # general running
    batch_size = 160
    feature_map = 130
    filter_hs = [3]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_.lexicon)
    """


def event_script2():
    trainbase = '../data/Event/English/sub_train.dat'
    translate_base = '../data/Event/translate/moses_train.dat'
    chtrainbase = '../data/Event/Chinese/rest_test.dat.new'
    testbase = '../data/Event/Chinese/sub_test.dat'
    validbase = '../data/Event/Chinese/validset'

    cdep = False
    dep = True
    phr = False
    lex = False
    hasmlphidden = False
    logprefix = '../exp/event_ch'
    filter_hss = [[3], [3, 4], [3, 4, 5]]
    config_ = config(chtrainbase, 'ch', testbase, validbase, lex, phr, cdep, logprefix, usefscore=True)
    word2vec.rundown_config(config_)
    process_qc.datasetConstructRundown_config(config_, dep)
    # cnn_model.structure_script(config_, filter_hss, hasmlphidden)

    # general running english
    batch_size = 170
    feature_map = 90
    filter_hs = [3, 4, 5]
    cnn_model.general_rundown(batch_size, feature_map, filter_hs, config_)


if __name__ == '__main__':
    event_script_mono()




