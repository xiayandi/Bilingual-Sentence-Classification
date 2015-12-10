__author__ = 'yandixia'
import codecs
import sys
import os
import xml.etree.ElementTree as ET
from util import readInDependencyTriples, getTreeStructure, getAncestors
import buildBilingualDict


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
        sentence = line.split('\t')[1].strip()
        coarse_label = labels.split(':')[0]
        fine_label = labels
        sentences.append(sentence)
        coarse_labels.append(coarse_label)
        fine_labels.append(fine_label)
    return [sentences, coarse_labels, fine_labels]


def _getChineseQCstructure(trainfile, output):
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
        sentence = line.split('\t')[1].strip()
        coarse_label = labels.split(':')[0]
        fine_label = labels
        sentences.append(sentence)
        coarse_labels.append(coarse_label)
        fine_labels.append(fine_label)
    return [sentences, coarse_labels, fine_labels]


def _getEnglishQCstructure(trainfile, output):
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


def coreNLPChineseSegment(formattedfile, segfile, coreNLPresultFile):
    """
    :func call coreNLP software to segment chinese corpus
    :param formattedfile: the formatted file in the same format in formatChineseQC()
    :param segfile: the output segmented corpus
    :param coreNLPresultFile: the parsing result corresponding to formattedfile
    :return: n/a
    """
    # output raw queries
    queris, clbls, flbls = get_chinese_raw_sentences_labels(formattedfile)

    # parsing the output of coreNLP
    tree = ET.parse(coreNLPresultFile)
    root = tree.getroot()
    segsents = []

    for sentence in root.iter('sentence'):
        words = []
        for token in sentence.iter('token'):
            word = token.find('word').text
            words.append(word)
        segsents.append(' '.join(words))

    print len(segsents)
    print len(queris)
    try:
        assert len(segsents) == len(queris)
    except AssertionError:
        with codecs.open('../exp/1111.txt', 'w', 'utf8') as writer:
            for sent in segsents:
                writer.write(sent + '\n')
        with codecs.open('../exp/2222.txt', 'w', 'utf8') as writer:
            for sent in queris:
                writer.write(sent + '\n')
        print 'wrong sentence number error: coreNLPChineseSegment()'
        sys.exit()

    seginstlines = []
    for i, sent in enumerate(segsents):
        seginstlines.append(flbls[i] + '\t' + sent + '\n')

    with codecs.open(segfile, 'w', 'utf-8') as writer:
        writer.writelines(seginstlines)


def lemmatize(formattedCorpusFile, coreNLPResultFile, outputFile):
    queries, clbls, flbls = get_english_raw_sentences_labels(formattedCorpusFile)

    # parsing the coreNLP result file
    tree = ET.parse(coreNLPResultFile)
    root = tree.getroot()
    lemmatizedsentences = []

    for sentence in root.iter('sentence'):
        words = []
        lemmas = []
        for token in sentence.iter('token'):
            word = token.find('word').text
            lemma = token.find('lemma').text
            lemmas.append(lemma)
            words.append(word)
            assert len(lemmas) == len(words)
        lemmatizedsentences.append(' '.join(words[:1] + lemmas[1:]))

    assert len(lemmatizedsentences) == len(flbls)

    lemmatized_inst_lines = []
    for i, sent in enumerate(lemmatizedsentences):
        lemmatized_inst_lines.append(flbls[i] + '\t' + sent + '\n')

    with open(outputFile, 'w') as writer:
        writer.writelines(lemmatized_inst_lines)


def coreNLPparser(formattedFile, lang, rawsentfile, outputdir):
    # output raw queries
    queris, clbls, flbls = get_chinese_raw_sentences_labels(formattedFile)

    with codecs.open(rawsentfile, 'w', 'utf-8') as writer:
        for q in queris:
            writer.write(q + '\n')

    if lang == 'ch':
        command = 'bash coreNLPchinese.sh ' + rawsentfile + ' ' + outputdir
        os.system(command)
    else:
        command = 'bash coreNLPenglish.sh ' + rawsentfile + ' ' + outputdir
        os.system(command)


def runCoreNLP(rawsentfile, lang):
    if lang == 'ch':
        command = 'bash coreNLPchinese.sh ' + rawsentfile
        os.system(command)
    else:
        command = 'bash coreNLPenglish.sh ' + rawsentfile
        os.system(command)


def outputBasicDependencyTriples(coreNLPParseFile, depTripleOutputFile):
    """
    :func extract basic dependency triples from coreNLP results and then output
            the triples to file
    :param coreNLPParseFile: the coreNLP result file
    :param depTripleOutputFile: triple output file
    :return: n/a
    """
    print 'extracting dependency triples from coreNLP result files...'
    tree = ET.parse(coreNLPParseFile)
    root = tree.getroot()
    sentences_deps = []
    for sentence in root.iter('sentence'):
        basicdep = sentence.find("dependencies[@type='basic-dependencies']")
        triples = ''
        for dep in basicdep.iter('dep'):
            dep_type = dep.attrib['type']
            dep_governor_idx = dep.find('governor').attrib['idx']
            dep_dependent_idx = dep.find('dependent').attrib['idx']
            triples += '(' + dep_governor_idx + ',' + dep_type + ',' + dep_dependent_idx + ')@'
        sentences_deps.append(triples.rstrip('@') + '\n')
    with codecs.open(depTripleOutputFile, 'w', 'utf-8') as writer:
        writer.writelines(sentences_deps)


def outputCollapsedDependencyTriples(coreNLPParseFile, depTripleOutputFile):
    """
    :func extract collapsed dependency triples from coreNLP results and then output
            the triples to file
    :param coreNLPParseFile: the coreNLP result file
    :param depTripleOutputFile: triple output file
    :return: n/a
    """
    print 'extracting dependency triples from coreNLP result files...'
    tree = ET.parse(coreNLPParseFile)
    root = tree.getroot()
    sentences_deps = []
    for sentence in root.iter('sentence'):
        collapseddep = sentence.find("dependencies[@type='collapsed-dependencies']")
        triples = ''
        for dep in collapseddep.iter('dep'):
            dep_type = dep.attrib['type']
            dep_governor_idx = dep.find('governor').attrib['idx']
            dep_dependent_idx = dep.find('dependent').attrib['idx']
            triples += '(' + dep_governor_idx + ',' + dep_type + ',' + dep_dependent_idx + ')@'
        sentences_deps.append(triples.rstrip('@') + '\n')
    with codecs.open(depTripleOutputFile, 'w', 'utf-8') as writer:
        writer.writelines(sentences_deps)


def word2phrase_sentencelevel(sentence, phrasemap):
    """
    func: transfer word based sentence into phrase based one
    :param sentence:  the sentence that needs to be transferred
    :param phrasemap: a phrase storing structure variable
    :return: the transferred sentence
    """
    ##words = [wd.lower() for wd in sentence.split()]+['PADDING']
    words = [wd.lower() for wd in sentence.split()] + ['PADDING']
    orig_words = sentence.split()
    orig_words[0] = orig_words[0].lower()
    newwords = []
    switch = True
    for i in xrange(len(words[:-1])):
        if switch:
            if words[i] in phrasemap:
                if words[i + 1] in phrasemap[words[i]]:
                    newwords.append(orig_words[i] + '_' + orig_words[i + 1])
                    switch = False
                else:
                    newwords.append(orig_words[i])
            else:
                newwords.append(orig_words[i])
        else:
            switch = True
    return ' '.join(newwords)


def mergeTwoNodesInTree(idx1, idx2, deptriples):
    assert idx1 == idx2-1
    g2d, d2g = getTreeStructure(deptriples)
    newTree = []
    vevset = set()
    for triple in deptriples:
        tup_1 = triple[0]
        tup_2 = triple[1]
        tup_3 = triple[2]
        if tup_1 > idx1:
            tup_1 -= 1
        if tup_3 > idx1:
            tup_3 -= 1
        if (tup_1, tup_3) not in vevset:
            vevset.add((tup_1, tup_3))
            newTree.append((tup_1, tup_2, tup_3))
    return newTree


def mergeDependencyTree(deptriples, sentence, phrasemap):
    """
    func: merge nodes that form phrases in the dependency tree
    :param deptriples: a dependency tree in the form of V.E.V. triple set
    :param sentence: corresponding word based sentence
    :param phrasemap: a phrase storing structure in buildingBilingualDict.py
    :return: a new set of dependency triples
    """
    # looking for phrases
    words = ['ROOT']+[wd.lower() for wd in sentence.split()]
    wordidxs = []
    switch = True
    for i in xrange(len(words[:-1])):
        if switch:
            if words[i] in phrasemap:
                if words[i + 1] in phrasemap[words[i]]:
                    wordidxs.append((i, i+1))
                    switch = False
        else:
            switch = True
    newTree = deptriples
    for i, (wdidx1, wdidx2) in enumerate(wordidxs):
        newTree = mergeTwoNodesInTree(wdidx1-i, wdidx2-i, newTree)
    return newTree


def formDependencyTripleLine(deptriplelist):
    depline = ''
    for triple in deptriplelist:
        depline += '('+str(triple[0])+','+triple[1]+','+str(triple[2])+')@'
    depline = depline.rstrip('@') + '\n'
    return depline


def word2phrase_filelevel(formatCorpusFile, depFile, outputPhraseCorpusFile, outputPhraseDepFile):
    """
    func: transfer word based corpus into phrase based one
    :param formatCorpusFile: the file that needs to be transfered
    :param depFile: the corresponding dependency file
    :param outputPhraseCorpusFile: the transfered phrase based output file
    :param outputPhraseDepFile: the transfered phrase based dependency file
    :return: n/a
    """
    phrasemap = buildBilingualDict.phraseMapping('../data/phrase.lst')
    sentences_triples = readInDependencyTriples(depFile)
    sentences, clbls, flbls = get_english_raw_sentences_labels(formatCorpusFile)
    assert len(sentences) == len(sentences_triples)

    newDepinfo = []
    newCorpus = []

    for i, sent in enumerate(sentences):
        sent_triples = sentences_triples[i]
        newdeps = mergeDependencyTree(sent_triples, sent, phrasemap)
        newsent = word2phrase_sentencelevel(sent, phrasemap)
        newDepinfo.append(formDependencyTripleLine(newdeps))
        newCorpus.append(flbls[i]+'\t'+newsent+'\n')

    with open(outputPhraseCorpusFile, 'w') as writer:
        writer.writelines(newCorpus)
    with open(outputPhraseDepFile, 'w') as writer:
        writer.writelines(newDepinfo)


def preprocess(corpusFile, lang):
    base_dir = '/'.join(corpusFile.split('/')[:-1]) + '/'
    fn = corpusFile.split('/')[-1]
    coreNLPparser(corpusFile, lang, corpusFile + '.sent', base_dir)

    if lang == 'ch':
        coreNLPChineseSegment(corpusFile, corpusFile + '.seg', corpusFile + '.sent.xml')

    outputBasicDependencyTriples(corpusFile + '.sent.xml', corpusFile + '.dep')


def rundown():
    corpusFileList = [
        # '../data/Semantic/eng_train',
        #'../data/Semantic/movieReview/trans_imdb/moses_trans_mr_eng2ch_train',
        #       '../data/QC/TREC/trimengqctrain',
        #       '../data/QC/Chinese_qc/finaltrain',
        #       '../data/QC/translate/final_moses_eng2ch_train',
        #       '../data/QC/Chinese_qc/finaltest'
    ]
    corpusLangList = ['ch']
    for i, corpusfoo in enumerate(corpusFileList):
        preprocess(corpusfoo, corpusLangList[i])
        if corpusLangList[i] == 'eng':
            word2phrase_filelevel(
                corpusfoo,
                corpusfoo + '.dep',
                corpusfoo + '.phr',
                corpusfoo + '.phr.dep'
            )


if __name__ == "__main__":
    #rundown()
    corpusfoo = '../data/Event/English/sub_train.dat'
    word2phrase_filelevel(
        corpusfoo,
        corpusfoo + '.dep',
        corpusfoo + '.phr',
        corpusfoo + '.phr.dep'
    )


