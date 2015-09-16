__author__ = 'yandixia'
import codecs
import sys


def formatChineseQC(rawfile, targetfile):
    with codecs.open(rawfile, 'r', 'utf-8') as reader:
        lines = reader.readlines()
    formatedlines = []
    for line in lines:
        items = line.split('\t')
        clabel, flabel = items[0].split('_')
        sentence = items[1].strip()
        if clabel[0] == u'\ufeff':
            clabel = clabel[1:]
            flabel = flabel
        if sentence[-1] != u'\uff1f':
            sentence += u'\uff1f'
        formatedlines.append(clabel + ':' + flabel + '\t' + sentence + '\n')
    with codecs.open(targetfile, 'w', 'utf-8') as writer:
        writer.writelines(formatedlines)


def formatTRECQC(rawfile, targetfile):
    with open(rawfile, 'r') as reader:
        lines = reader.readlines()
    formatedlines = []
    for line in lines:
        items = line.split()
        labels = items[0]
        sentence = ' '.join(items[1:])
        formatedlines.append(labels + '\t' + sentence + '\n')
    with open(targetfile, 'w') as writer:
        writer.writelines(formatedlines)


def get_chinese_raw_sentences_labels(datafile):
    """
    function: read in the datafile, and separate it into sentences,
            coarse labels and fine labels
    :param datafile: a data file with the format
            "clbl:flbl\tword1 word2 word3 word4 word5 word6 word7 ?\tprojectName"
    :return: a list of lists, ie. [sentence list, clbl list, flbl list]
    """
    sentences = []
    coarse_labels = []
    fine_labels = []

    with codecs.open(datafile, 'r', 'utf-8') as reader:
        datalines = reader.readlines()
    for line in datalines:
        labels = line.split('\t')[0]
        sentence = line.split('\t')[1]
        coarse_label = labels.split(':')[0]
        fine_label = labels
        sentences.append(sentence)
        coarse_labels.append(coarse_label)
        fine_labels.append(fine_label)
    return [sentences, coarse_labels, fine_labels]


def getChineseQCstructure(trainfile, output):
    """
    function: given training file find the label structure automatically,
            and output it
    :param train_file: training file
    :param output: the file to save the corresponding train file
    :return: None
    """
    sents, clbls, flbls = get_chinese_raw_sentences_labels(trainfile)

    struct = {}

    for i, clbl in enumerate(clbls):
        if clbl not in struct:
            struct[clbl] = set()
        struct[clbl].add(flbls[i])

    with codecs.open(output, 'w', 'utf-8') as writer:
        for clbl in struct:
            cflbls = struct[clbl]
            writer.write(clbl + '\n')
            for cflbl in cflbls:
                writer.write(cflbl + '\n')
            writer.write('\n')


def get_english_raw_sentences_labels(datafile):
    """
    function: read in the datafile, and separate it into sentences,
            coarse labels and fine labels
    :param datafile: a data file with the format
            "clbl:flbl\tword1 word2 word3 word4 word5 word6 word7 ?\tprojectName"
    :return: a list of lists, ie. [sentence list, clbl list, flbl list]
    """
    sentences = []
    coarse_labels = []
    fine_labels = []

    with open(datafile, 'r') as reader:
        datalines = reader.readlines()
    for line in datalines:
        labels = line.split('\t')[0]
        sentence = line.split('\t')[1]
        coarse_label = labels.split(':')[0]
        fine_label = labels
        sentences.append(sentence)
        coarse_labels.append(coarse_label)
        fine_labels.append(fine_label)
    return [sentences, coarse_labels, fine_labels]


def getEnglishQCstructure(trainfile, output):
    """
    function: given training file find the label structure automatically,
            and output it
    :param train_file: training file
    :param output: the file to save the corresponding train file
    :return: None
    """
    sents, clbls, flbls = get_english_raw_sentences_labels(trainfile)

    struct = {}

    for i, clbl in enumerate(clbls):
        if clbl not in struct:
            struct[clbl] = set()
        struct[clbl].add(flbls[i])

    with open(output, 'w') as writer:
        for clbl in struct:
            cflbls = struct[clbl]
            writer.write(clbl + '\n')
            for cflbl in cflbls:
                writer.write(cflbl + '\n')
            writer.write('\n')


def getLabelMapping():
    with open('../data/QC/label_mapping', 'r') as reader:
        lmlines = reader.readlines()

    lmidx = 0
    clblmap = {}
    for i, line in enumerate(lmlines):
        if line[0] == '#':
            continue
        elif line.strip():
            chlbl, treclbl = line.split()
            clblmap[chlbl] = treclbl
        else:
            lmidx = i + 1
            break

    flblmap = {}
    for i, line in enumerate(lmlines[lmidx:]):
        if line[0] == '#':
            continue
        elif line.strip():
            chlbl, treclbl = line.split()
            flblmap[chlbl] = treclbl
        else:
            lmidx += i + 1
            break

    delset = set()
    for i, line in enumerate(lmlines[lmidx:]):
        if line[0] == '#':
            continue
        elif line.strip():
            delset.add(line.strip())
        else:
            break
    return clblmap, flblmap, delset


def trimChineseQC(formatedchfile, trimmedfile):
    """
    :func delete unwanted labels and map chinese labels to TREC labels
    :param formatedchfile: formatted chinese QC corpus using formatChineseQC() function
    :param trimmedfile: the output trimmed file
    :return: n/a
    """
    clblmap, flblmap, delset = getLabelMapping()

    with codecs.open(formatedchfile, 'r', 'utf-8') as reader:
        instlines = reader.readlines()

    trimmedinstlines = []

    for line in instlines:
        items = line.split('\t')
        clbl = items[0].split(':')[0]
        flbl = items[0]
        query = items[1]
        if clbl in clblmap:
            clbl = clblmap[clbl]
            flbl = clbl + ':' + items[0].split(':')[1]
        if flbl in flblmap:
            flbl = flblmap[flbl]
        if flbl not in delset:
            trimmedinstlines.append(flbl + '\t' + query)
    with codecs.open(trimmedfile, 'w', 'utf-8') as writer:
        writer.writelines(trimmedinstlines)


def rundown():
    formatChineseQC('../data/QC/Chinese_qc/trainutf8.txt', '../data/QC/Chinese_qc/formatTrain')
    formatChineseQC('../data/QC/Chinese_qc/testutf8.txt', '../data/QC/Chinese_qc/formatTest')
    formatTRECQC('../data/QC/TREC/train_5500.label', '../data/QC/TREC/formatTrain')
    formatTRECQC('../data/QC/TREC/TREC_10.label', '../data/QC/TREC/formatTest')

    trimChineseQC('../data/QC/Chinese_qc/formatTrain', '../data/QC/Chinese_qc/trimchqctrain')
    trimChineseQC('../data/QC/Chinese_qc/formatTest', '../data/QC/Chinese_qc/trimchqctest')

    getChineseQCstructure('../data/QC/Chinese_qc/trimchqctrain', '../data/QC/Chinese_qc/label_struct')
    getEnglishQCstructure('../data/QC/TREC/formatTrain', '../data/QC/TREC/label_struct')


if __name__ == "__main__":
    rundown()

