__author__ = 'yandixia'
import codecs
import sys
import os
import xml.etree.ElementTree as ET
from util import readInDependencyTriples, getTreeStructure, getAncestors
import buildBilingualDict


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
        sentence = line.split('\t')[1].strip()
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
        sentence = line.split('\t')[1].strip()
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


def trimEnglishQC(formatedengfile, trimmedfile):
    """
    :func delete unwanted labels and map chinese labels to TREC labels
    :param formatedchfile: formatted chinese QC corpus using formatChineseQC() function
    :param trimmedfile: the output trimmed file
    :return: n/a
    """
    delset = {'ABBR:exp', 'LOC:state', 'LOC:other', 'HUM:title', 'NUM:other', 'ENTY:product',
              'ENTY:techmeth', 'ENTY:termeq', 'ENTY:word', 'ENTY:letter', 'ENTY:other', 'ENTY:veh',
              'ENTY:instru'}

    with open(formatedengfile, 'r') as reader:
        instlines = reader.readlines()

    trimmedinstlines = []

    for line in instlines:
        items = line.split('\t')
        flbl = items[0]
        query = items[1]
        if flbl not in delset:
            trimmedinstlines.append(flbl + '\t' + query)
    with open(trimmedfile, 'w') as writer:
        writer.writelines(trimmedinstlines)


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
    assert len(segsents) == len(queris)

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


def coreNLPparser(formattedFile, lang, rawsentfile):
    # output raw queries
    queris, clbls, flbls = get_chinese_raw_sentences_labels(formattedFile)

    with codecs.open(rawsentfile, 'w', 'utf-8') as writer:
        for q in queris:
            writer.write(q + '\n')

    if lang == 'ch':
        command = 'bash coreNLPchinese.sh ' + rawsentfile
        os.system(command)
    else:
        command = 'bash coreNLPenglish.sh ' + rawsentfile
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


def rundown():
    formatChineseQC('../data/QC/Chinese_qc/trainutf8.txt', '../data/QC/Chinese_qc/formatTrain')
    formatChineseQC('../data/QC/Chinese_qc/testutf8.txt', '../data/QC/Chinese_qc/formatTest')
    formatTRECQC('../data/QC/TREC/train_5500.label', '../data/QC/TREC/formatTrain')
    formatTRECQC('../data/QC/TREC/TREC_10.label', '../data/QC/TREC/formatTest')

    trimChineseQC('../data/QC/Chinese_qc/formatTrain', '../data/QC/Chinese_qc/trimchqctrain')
    trimChineseQC('../data/QC/Chinese_qc/formatTest', '../data/QC/Chinese_qc/trimchqctest')
    trimEnglishQC('../data/QC/TREC/formatTrain', '../data/QC/TREC/trimengqctrain')
    trimEnglishQC('../data/QC/TREC/formatTest', '../data/QC/TREC/trimengqctest')

    coreNLPparser('../data/QC/Chinese_qc/trimchqctrain', 'ch', '../exp/ch_qc_train')
    coreNLPparser('../data/QC/Chinese_qc/trimchqctest', 'ch', '../exp/ch_qc_test')
    coreNLPparser('../data/QC/TREC/trimengqctrain', 'eng', '../exp/eng_qc_train')
    coreNLPparser('../data/QC/TREC/trimengqctest', 'eng', '../exp/eng_qc_test')

    coreNLPChineseSegment('../data/QC/Chinese_qc/trimchqctrain',
                          '../data/QC/Chinese_qc/finaltrain',
                          '../exp/ch_qc_train.xml')
    coreNLPChineseSegment('../data/QC/Chinese_qc/trimchqctest',
                          '../data/QC/Chinese_qc/finaltest',
                          '../exp/ch_qc_test.xml')

    # lemmatize('../data/QC/TREC/trimengqctrain', '../exp/eng_qc_train.xml', '../data/QC/TREC/lemmaFormatTrain')
    #lemmatize('../data/QC/TREC/trimengqctest', '../exp/eng_qc_test.xml', '../data/QC/TREC/lemmaFormatTest')

    outputBasicDependencyTriples('../exp/ch_qc_train.xml', '../exp/ch_qc_train_dep')
    outputBasicDependencyTriples('../exp/ch_qc_test.xml', '../exp/ch_qc_test_dep')
    outputBasicDependencyTriples('../exp/eng_qc_train.xml', '../exp/eng_qc_train_dep')
    outputBasicDependencyTriples('../exp/eng_qc_test.xml', '../exp/eng_qc_test_dep')


def word2phrase_sentencelevel(sentence, phrasemap):
    """
    func: transfer word based sentence into phrase based one
    :param sentence:  the sentence that needs to be transferred
    :param phrasemap: a phrase storing structure variable
    :return: the transferred sentence
    """
    words = sentence.split()
    newwords = []
    switch = True
    for i in xrange(len(words[:-1])):
        if switch:
            if words[i] in phrasemap:
                if words[i + 1] in phrasemap[words[i]]:
                    newwords.append(words[i] + '_' + words[i + 1])
                    switch = False
                else:
                    newwords.append(words[i])
            else:
                newwords.append(words[i])
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
    words = sentence.split()
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
        depline += '('+triple[0]+','+triple[1]+','+triple[2]+')@'
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
        newCorpus.append(flbls+'\t'+newsent+'\n')

    with open(outputPhraseCorpusFile, 'w') as writer:
        writer.writelines(newCorpus)
    with open(outputPhraseDepFile, 'w') as writer:
        writer.writelines(newDepinfo)


def preprocessTrans(transfile):
    """
    func: adding Chinese format question mark at the end
    :param transfile: the translated file by Google translation
    :return: n/a
    """
    with codecs.open(transfile, 'r', 'utf-8') as reader:
        lines = reader.readlines()
    qm = u'\uff1f'  # chinese question mark
    for i, line in enumerate(lines):
        lines[i] = line[:-2].replace(qm, '') + qm + '\n'
    for i, line in enumerate(lines):
        if line.strip()[-1] != qm:
            lines[i] = line.strip()[:-1] + qm + '\n'
    with codecs.open(transfile, 'w', 'utf-8') as writer:
        writer.writelines(lines)


def rundownOnTranslate():
    runCoreNLP('../data/QC/translate/google_ch2eng_test', 'eng')
    runCoreNLP('../data/QC/translate/google_eng2ch_train', 'ch')

    coreNLPChineseSegment('../data/QC/TREC/trimengqctrain',
                          '../data/QC/translate/final_google_eng2ch_train',
                          '../exp/google_eng2ch_train.xml')
    coreNLPChineseSegment('../data/QC/Chinese_qc/trimchqctest',
                          '../data/QC/Chinese_qc/final_google_ch2eng_test',
                          '../exp/google_ch2eng_test.xml')

    outputBasicDependencyTriples('../exp/google_eng2ch_train.xml', '../exp/google_eng2ch_train_dep')
    outputBasicDependencyTriples('../exp/google_ch2eng_test.xml', '../exp/google_ch2eng_test_dep')


if __name__ == "__main__":
    #rundownOnTranslate()
    word2phrase_filelevel('../data/QC/TREC/trimengqctrain', '../exp/eng_qc_train_dep', '../data/QC/TREC/phraseengtrain',
                          '../exp/phrase_eng_qc_train_dep')


