"""
this module is for test only
"""
import codecs
import xml.etree.ElementTree as ET
import sys


def outputDependencyTriples(coreNLPParseFile, depTripleOutputFile):
    """
    :func extract dependency triples from coreNLP results and then output
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


def readInDependencyTriples(depTripleFile):
    """
    :func read dependency triples into memory from triple file
    :param depTripleFile: dependency triple file
    :return: a sentence level list of dependency triples
    """
    with open(depTripleFile, 'r') as reader:
        triplelines = reader.readlines()
    sentences_triples = []
    for line in triplelines:
        triples = []
        str_triples = line.rstrip().split('@')
        for str_triple in str_triples:
            g, t, d = str_triple.strip('()').split(',')  # g:governor, t:type, d:dependent
            triples.append((int(g), t, int(d)))
        sentences_triples.append(triples)
    return sentences_triples


def getTreeStructure(sentTriples):
    """
    :func get a very simple V.E.V. fashion tree structure, to easily get
            ancestors and siblings
    :param sentTriples: dependency triples of a sentence
    :return: g2d: governor to dependent d2g: dependent to governor
    """
    g2d = dict()  # governor to dependent
    d2g = dict()  # dependent to governor
    for triple in sentTriples:
        if triple[0] in g2d:
            g2d[triple[0]].append(triple[2])
        else:
            g2d[triple[0]] = [triple[2]]
        if triple[2] in d2g:
            print 'hello'
            d2g[triple[2]].append(triple[0])
        else:
            d2g[triple[2]] = [triple[0]]
    return g2d, d2g


def testRundown():
    sentences_triples = readInDependencyTriples('../exp/deps')
    for sent_triple in sentences_triples:
        getTreeStructure(sent_triple)


if __name__ == '__main__':
    print 'test'
    testRundown()
