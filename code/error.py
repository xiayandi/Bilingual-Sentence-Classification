__author__ = 'yandixia'

import process_qc


def findErrors(predictionFile, goldFile, testCorpusFile, label_vector):
    """
    func: find all the errors predicted by the model, and print them in human readable way
    :param predictionFile: the output prediction from cnn_model.py, a sequence of label indexes
    :param goldFile: the gold label sequence output by cnn_model.py
    :param testCorpusFile: the formatted test corpus file, used to retrieve actual text data
    :return: n/a
    """
    # read in prediction file
    with open(predictionFile, 'r') as reader:
        predslines = reader.readlines()
    # read in gold file
    with open(goldFile, 'r') as reader:
        goldlines = reader.readlines()
    # read in formatted test corpus file
    with open(testCorpusFile, 'r') as reader:
        textlines = reader.readlines()

    errorPredsInsts = []
    for i in xrange(len(predslines)):
        if predslines[i] != goldlines[i]:
            errorPredsInsts.append(label_vector[int(predslines[i])] + '\t' + textlines[i])

    errorFile = '../exp/error_analysis'
    with open(errorFile, 'w') as writer:
        writer.writelines(errorPredsInsts)


def errorAnalysisRundown():
    clbl_vec, flbl_vec = process_qc.label_structure('../exp/label_struct_bi')
    predsFile = '../exp/predictions'
    goldFile = '../exp/goldrs'
    textFile = '../data/QC/Chinese_qc/finaltest'
    findErrors(predsFile, goldFile, textFile, clbl_vec)


if __name__ == '__main__':
    errorAnalysisRundown()
