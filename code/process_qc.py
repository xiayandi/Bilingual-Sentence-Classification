__author__ = 'yandixia'

import numpy as np
import operator
import cPickle
import codecs
from util import readInDependencyTriples, getTreeStructure, getAncestors, myGetAncestors
import sys


def constructDependencyBasedData(ancestornum, corpusFile, depTripleFile, outputFile):
    """
    :func construct dependency based data. the normal sentences are reordered
            using dependency information
    :param ancestornum: how many ancestors to append after the word
    :param corpusFile: formatted corpus file with normal sentences
    :param depTripleFile: the dependency triple file extracted from coreNLP result
    :param outputFile: the output file that stores the dependency based data
    :return: n/a
    """
    sentences, c_lbls, f_lbls = get_raw_sentences_labels(
        corpusFile
    )
    sentences_triples = readInDependencyTriples(depTripleFile)

    assert len(sentences) == len(sentences_triples)

    reordered_sentences = []
    for i, sent in enumerate(sentences):
        g2d, d2g = getTreeStructure(sentences_triples[i])
        words = ['PADDING'] + sent.split()
        windows = []
        for j in xrange(1, len(words)):
            ancestors = myGetAncestors(j, ancestornum, d2g)
            if ancestors is not None:
                windows.append(' '.join([words[k] for k in [j] + ancestors]))
        reordered_sentences.append(' '.join(windows))
    assert len(reordered_sentences) == len(sentences)

    newinstances = [
        f_lbls[i] + '\t' + reordered_sentences[i] + '\n'
        for i in xrange(len(reordered_sentences))
    ]

    with open(outputFile, 'w') as writer:
        writer.writelines(newinstances)


def get_idx_from_dep_pattern(deppatterns, word_idx_map, max_l, filter_h):
    """
    function: given a set of dependency based patterns, constructed by
            constructDependencyBasedData, return a index sentence with
            each index represents the word index in the word embedding dictionary
    :param sent: text sentence
    :param word_idx_map: a hash variable mapping text word to int index
    :param max_l: the global max length of the sentences in training data
    :param filter_h: window size
    :return: a vector with each element representing a word index in the sentence
    """
    # Transforms dependency pattern into a list of indices.

    x = []
    words = deppatterns.split()
    assert len(words) % filter_h == 0

    windows = (words[i:i + filter_h] for i in xrange(0, len(words), filter_h))
    for i, window in enumerate(windows):
        window_x = []
        for word in window:
            if word in word_idx_map:  # remove unkown words
                window_x.append(word_idx_map[word])
            else:
                window_x.append(0)
        x.append(window_x)
    while len(x) < max_l:
        x.append([0] * filter_h)
    return x


def get_idx_from_sent(sent, word_idx_map, max_l, filter_h):
    """
    function: given a text sentence, return a index sentence with each index represents
            the word index in the word embedding dictionary
    :param sent: text sentence
    :param word_idx_map: a hash variable mapping text word to int index
    :param max_l: the global max length of the sentences in training data
    :param filter_h: window size
    :return: a vector with each element representing a word index in the sentence
    """
    # Transforms sentence into a list of indices. Pad with zeroes.
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:  # remove unkown words
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def find_max_length(sentences):
    """
    function: find the max length of the given list of sentences
    :param sentences: a list of sentences
    :return: the max length(int)
    """
    max_length = 0
    for sentence in sentences:
        if len(sentence.split()) > max_length:
            max_length = len(sentence.split())
    return max_length


def get_raw_sentences_labels(datafile):
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
        sentence = line.split('\t')[1].rstrip()
        coarse_label = labels.split(':')[0]
        fine_label = labels
        sentences.append(sentence)
        coarse_labels.append(coarse_label)
        fine_labels.append(fine_label)
    return [sentences, coarse_labels, fine_labels]


def output_label_structure(train_file, output):
    """
    function: given training file find the label structure automatically,
            and output it
    :param train_file: training file
    :param output: the file to save the corresponding train file
    :return: None
    """
    clbls, flbls = get_raw_sentences_labels(train_file)[1:]

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


def get_w_to_idx_map(vocab_file):
    """
    function: generate a mapping from word to its index in vocab list
    :param vocab_file: a fixed vocab list file
    :return: a hash map from word to index
    """
    print 'producing word to index mapping...'
    w2idxmap = {}
    with open(vocab_file, 'r') as reader:
        vocablines = reader.readlines()

    for i in xrange(len(vocablines)):
        line = vocablines[i]
        wd = line.split()[0]
        assert wd not in w2idxmap
        w2idxmap[wd] = i
    print 'producing word to index mapping done!'
    return w2idxmap


def label_structure(structureFile):
    """
    function: get coarse label list and fine label list
    :param structureFile: the structure file that is output by output_label_structure()
    :return: coarse label list and fine label list
    """
    with open(structureFile, 'r') as reader:
        lines = reader.readlines()
    prev = ''
    clbl_vec = []
    flbl_vec = []
    for line in lines:
        if line.strip():
            if prev == '':
                current_c = line.strip()
                clbl_vec.append(current_c)
            else:
                flbl_vec.append(line.strip())
        prev = line.strip()
    return clbl_vec, flbl_vec


def get_lbl_to_idx_map(label_stuct_file):
    """
    functioin: get hash map from label to index in the label lists output by label_structure()
    :param label_stuct_file: label structure file output by output_label_structure()
    :return: coarse label and fine label to index hash map
    """
    c_lbls, f_lbls = label_structure(label_stuct_file)
    c_map = {}
    f_map = {}
    for i in xrange(len(c_lbls)): c_map[c_lbls[i]] = i
    for i in xrange(len(f_lbls)): f_map[f_lbls[i]] = i

    return c_map, f_map


def getVocab(vocabfile):
    """
    function: get vobabulary set
    :param vocabfile: the vocabulary file output by word2vec.py
    :return: vocabulary set
    """
    with open(vocabfile, 'r') as reader:
        vocablines = reader.readlines()
    vocabset = set()
    for line in vocablines:
        vocabset.add(line.strip())
    return vocabset


def get_length_mask(sentences):
    """
    function: get a sentence length list with each element representing
            length of the corresponding sentence
    :param sentences: a list of sentences
    :return: a list of sentences
    """
    mask = []
    for sent in sentences:
        mask.append(len(sent.split()))
    return mask


def find_global_max_length(filelist):
    """
    func: find the max sentence length across a list of corpus files
    :param filelist: a list of corpus files
    :return: the max length
    """
    max_l = 0
    for foo in filelist:
        sentences, c_lbls, f_lbls = get_raw_sentences_labels(
            foo
        )
        length = find_max_length(sentences)
        if length > max_l:
            max_l = length
    return max_l


def construct_additional_features(trainfile, testfile, featurelist='featurelist'):
    with open(featurelist, 'r') as reader:
        featurelines = reader.readlines()
    featvocab = set([line.strip() for line in featurelines])

    additional_data = []
    for datafile in [trainfile, testfile]:
        sentences, c_lbls, f_lbls = get_raw_sentences_labels(
            datafile
        )
        additional_features = []
        for sent in sentences:
            words = sent.split()
            hit = False
            for word in words:
                if word in featvocab:
                    hit = True
            if hit:
                additional_features.append(1.0)
            else:
                additional_features.append(0.0)
        np_add_feats = np.asarray(additional_features, dtype=np.float32)
        print np_add_feats.shape
        additional_data.append(np_add_feats)
    dataset_output = open('../exp/addfeats', 'wb')
    cPickle.dump(additional_data, dataset_output, -1)
    dataset_output.close()


def construct_dataset(datafile, filter_h, max_l, lbl2idxmap, vocab_file, indexizer):
    """
    function: convert text version data into index version used by CNN model.
    :param datafile: training, test or valid data file
    :param filter_h: window size
            ############## ATTENTION #################
            This is the right parameter that you should change, if you want
            to experiment with different window size.
    :param lbl2idxmap: label to index hash map returned by get_lbl_to_idx_map()
    :param vocab_file: the vocabulary file produced in word2vec.py
    :return: a list containing index matrix for all the sentences in the datafile,
            coarse label vector for all the sentences, fine label vector for all
            the sentences and length vector for all the sentences
    """
    sentences, c_lbls, f_lbls = get_raw_sentences_labels(
        datafile
    )

    w2idx = get_w_to_idx_map(vocab_file)

    c2idx, f2idx = lbl2idxmap

    sent_idx_seq = []
    c_idx_seq = []
    f_idx_seq = []
    for i in xrange(len(sentences)):
        sent = sentences[i]
        c_lbl = c_lbls[i]
        f_lbl = f_lbls[i]
        sent_idx = indexizer(sent, w2idx, max_l, filter_h)
        c_idx = c2idx[c_lbl]
        f_idx = f2idx[f_lbl]
        sent_idx_seq.append(sent_idx)
        c_idx_seq.append(c_idx)
        f_idx_seq.append(f_idx)

    # sent_array = np.array(sent_idx_seq, dtype="int32")
    #c_array = np.array(c_idx_seq, dtype="int32")
    #f_array = np.array(f_idx_seq, dtype="int32")
    #mask_array = np.array(get_length_mask(sentences), dtype="int32")
    #return [sent_array, c_array, f_array, mask_array]
    return [sent_idx_seq, c_idx_seq, f_idx_seq]


def convertNumpy(lst):
    datasets = []
    for v in lst:
        sent_array = np.array(v[0], dtype="int32")
        c_array = np.array(v[1], dtype="int32")
        f_array = np.array(v[2], dtype="int32")
        dataset = [sent_array, c_array, f_array]
        datasets.append(dataset)
    return datasets


def mixCorpus(eng_part, ch_part, eng_proportion, ch_proportion):
    assert len(eng_part) == len(ch_part) == 3
    mixed = []
    for i in xrange(len(eng_part)):
        subpart = eng_part[i][:eng_proportion * len(eng_part[0]) / 10] + ch_part[i][
                                                                         :ch_proportion * len(ch_part[0]) / 10]
        mixed.append(subpart)
    return mixed


def permute(data):
    """
    func: permute construct_dataset() output format data
    :param data: data in the format of construct_dataset()
    """
    assert len(data) == 3
    rng = np.random.RandomState(1234)
    indexes = rng.permutation(range(len(data[0])))
    data[0] = [data[0][i] for i in indexes]
    data[1] = [data[1][i] for i in indexes]
    data[2] = [data[2][i] for i in indexes]


def datasetConstructRundown(eng_proportion, ch_proportion):
    """
    This is a demo script for showing how to use the defined functions to produce data
    that is required by CNN model.
    """
    #eng_train_file = '../data/QC/translate/final_google_eng2ch_train'  # '../data/QC/TREC/formatTrain'  # English training file original order

    #eng_train_file = '../data/QC/TREC/trimengqctrain'  # '../data/QC/TREC/formatTrain'  # English training file original order

    # eng_train_file = '../data/QC/TREC/phraseengtrain'  # '../data/QC/TREC/formatTrain'  # English training file original order

    #eng_train_file_base = '../data/QC/translate/final_moses_eng2ch_train'

    # eng_train_file_base = '../data/QC/TREC/trimengqctrain' # QC english train file
    #ch_test_file_base = '../data/QC/Chinese_qc/finaltest' # QC Chinese test file

    ###########################################
    #           English Train files           #
    ###########################################
    eng_train_file_base = '../data/Semantic/eng_train'
    eng_train_file = eng_train_file_base + '.phr'  # '../data/QC/TREC/formatTrain'  # English training file original order
    eng_train_dep_file = eng_train_file_base + '.phr.dep'  # English dependency triple file
    eng_tree_based_train_file = '../exp/eng_train.dat'  # English dependency based training file

    ###########################################
    # Chinese Train files           #
    ###########################################
    hasChineseTrain = False
    if hasChineseTrain:
        ch_train_file_base = '../data/QC/Chinese_qc/finaltrain'
        ch_train_file = ch_train_file_base + '.seg'
        ch_train_dep_file = ch_train_file_base + '.dep'
        ch_tree_based_train_file = '../exp/ch_train.dat'

    ###########################################
    # Chinese Test files            #
    ###########################################
    ch_test_file_base = '../data/Semantic/Douban/test.dat'
    ch_test_file = ch_test_file_base + '.seg'
    ch_test_dep_file = ch_test_file_base + '.dep'
    ch_tree_based_test_file = '../exp/test.dat'

    label_struct_file = '../exp/label_struct_bi'
    vocab_file = '../exp/vocab_bi.lst'
    outputDataFile = '../exp/dataset_bi.pkl'
    DepBased = True

    # output label structure file and get the label to index hash map
    output_label_structure(eng_train_file, label_struct_file)
    lbl2idxmap = get_lbl_to_idx_map(label_struct_file)

    # find the global max sentence length
    max_l = find_global_max_length([eng_train_file, ch_test_file])

    if DepBased:
        # reorder sentences
        ancestorNum = 4  # max window size is 4+1
        filter_h = ancestorNum + 1  # window size

        constructDependencyBasedData(ancestorNum, eng_train_file, eng_train_dep_file, eng_tree_based_train_file)
        if hasChineseTrain:
            constructDependencyBasedData(ancestorNum, ch_train_file, ch_train_dep_file, ch_tree_based_train_file)
        constructDependencyBasedData(ancestorNum, ch_test_file, ch_test_dep_file, ch_tree_based_test_file)

        # actual stage for constructing CNN data
        eng_train_part = construct_dataset(eng_tree_based_train_file, filter_h, max_l, lbl2idxmap, vocab_file,
                                           get_idx_from_dep_pattern)
        if hasChineseTrain:
            ch_train_part = construct_dataset(ch_tree_based_train_file, filter_h, max_l, lbl2idxmap, vocab_file,
                                              get_idx_from_dep_pattern)
        ch_test_part = construct_dataset(ch_tree_based_test_file, filter_h, max_l, lbl2idxmap, vocab_file,
                                         get_idx_from_dep_pattern)

    else:
        filter_h = 3
        # actual stage for constructing CNN data
        eng_train_part = construct_dataset(eng_train_file, filter_h, max_l, lbl2idxmap, vocab_file, get_idx_from_sent)
        if hasChineseTrain:
            ch_train_part = construct_dataset(ch_train_file, filter_h, max_l, lbl2idxmap, vocab_file, get_idx_from_sent)
        ch_test_part = construct_dataset(ch_test_file, filter_h, max_l, lbl2idxmap, vocab_file, get_idx_from_sent)

    # mix eng and ch train
    if hasChineseTrain:
        ch_proportion = ch_proportion
        eng_proportion = eng_proportion
        permute(ch_train_part)
        train_part = mixCorpus(eng_train_part, ch_train_part, eng_proportion, ch_proportion)
        test_part = ch_test_part
    else:
        train_part = eng_train_part
        test_part = ch_test_part

    dataset = convertNumpy([train_part, test_part])
    # uncomment next line if you have valid set
    # dataset = [train_part, test_part, valid_part]

    # display dataset info
    print 'basic info:'
    print 'max length is: ' + str(dataset[0][0].shape[1])
    print 'training set size: ' + str(dataset[0][0].shape)
    print 'test set size: ' + str(dataset[1][0].shape)
    construct_additional_features(eng_train_file, ch_test_file)

    dataset_output = open(outputDataFile, 'wb')
    cPickle.dump(dataset, dataset_output, -1)
    dataset_output.close()


if __name__ == '__main__':
    datasetConstructRundown(10, 0)

