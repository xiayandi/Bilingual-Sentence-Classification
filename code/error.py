__author__ = 'yandixia'

import process_qc
import numpy as np


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


def findCorrects(predictionFile, goldFile, testCorpusFile, label_vector):
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

    correctPredsInsts = []
    for i in xrange(len(predslines)):
        if predslines[i] == goldlines[i]:
            correctPredsInsts.append(label_vector[int(predslines[i])] + '\t' + textlines[i])

    correctFile = '../exp/correct_analysis'
    with open(correctFile, 'w') as writer:
        writer.writelines(correctPredsInsts)


def compareErrors(errfile1, errfile2):
    with open(errfile1, 'r') as reader:
        err1lines = reader.readlines()
    with open(errfile2, 'r') as reader:
        err2lines = reader.readlines()
    err2set = set([line.split('\t')[-1] for line in err2lines])
    count = 0
    print 'in err1 not in err2'
    for line in err1lines:
        if line.split('\t')[-1] not in err2set:
            count += 1
            print line.rstrip()
    print count


def label_distribution():
    with open('../data/QC/Chinese_qc/finaltest', 'r') as reader:
        lines = reader.readlines()
    label_dict = {}
    for line in lines:
        labels = line.split('\t')[0]
        clabel = labels.split(':')[0]
        if clabel in label_dict:
            label_dict[clabel] += 1
        else:
            label_dict[clabel] = 1
    for key, val in label_dict.items():
        print key + ' ' + str(val)


def duplicate_abbr():
    with open('../data/QC/TREC/trimengqctrain', 'r') as reader:
        lines = reader.readlines()
    with open('../exp/eng_qc_train_dep') as reader:
        deplines = reader.readlines()
    assert len(lines) == len(deplines)
    duplines = []
    dupdeplines = []
    for i, line in enumerate(lines):
        labels = line.split('\t')[0]
        clabel = labels.split(':')[0]
        if clabel == 'ABBR':
            duplines.extend([line] * 10)
            dupdeplines.extend([deplines[i]] * 10)
        else:
            duplines.append(line)
            dupdeplines.append(deplines[i])

    rng = np.random.RandomState(1234)
    indexes = rng.permutation(range(len(duplines)))
    permutelines = [duplines[i] for i in indexes]
    permutedeplines = [dupdeplines[i] for i in indexes]

    with open('../exp/duptrain', 'w') as writer:
        writer.writelines(permutelines)

    with open('../exp/dupdeptrain', 'w') as writer:
        writer.writelines(permutedeplines)


def sortTrecTrain():
    trectrainfile = '../data/QC/TREC/lemmaFormatTrain'
    sortedtrectrainfile = '../exp/sortedlemmatrectrain'
    with open(trectrainfile, 'r') as reader:
        trectrainlines = reader.readlines()
    with open(sortedtrectrainfile, 'w') as writer:
        writer.writelines(sorted(trectrainlines))


def errorAnalysisRundown():
    clbl_vec, flbl_vec = process_qc.label_structure('../exp/label_struct_bi')
    predsFile = '../exp/predictions'
    goldFile = '../exp/goldrs'
    textFile = '../data/QC/Chinese_qc/finaltest'
    findErrors(predsFile, goldFile, textFile, clbl_vec)
    findCorrects(predsFile, goldFile, textFile, clbl_vec)


if __name__ == '__main__':
    errorAnalysisRundown()
    # sortTrecTrain()
    # compareErrors('../exp/error_analysis1', '../exp/error_analysis')
