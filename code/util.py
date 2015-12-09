__author__ = 'yandixia'

"""
This is the util module for some basic operations
"""

import sys


def insertDict(key, dictionary):
    """
    func: insert into dictionary. Add one to the key entry
    param: key: the inserted key
    param: dictionary: the inserted dictionary
    return: n/a
    """
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


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
            try:
                g, t, d = str_triple.strip('()').split(',')  # g:governor, t:type, d:dependent
            except ValueError:
                print str_triple
                print depTripleFile
                sys.exit()
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
            d2g[triple[2]].append(triple[0])
        else:
            d2g[triple[2]] = [triple[0]]
    return g2d, d2g


def getAncestors(idx, level, d2g):
    """
    :func find a list of ancestors for the given idx
    :param idx: node idx
    :param level: how many ancestors to find, level 1 return node idx's parent
    :param g2d: governor to dependent mapping
    :param d2g: dependent to governor mapping
    :return: the list of ancestors if level is fulfilled, otherwise None
    """
    if level == 1:
        if idx in d2g:
            return [d2g[idx][0]]
        elif idx == 0:  # current node is ROOT
            return [0]
        else:
            return [0]
    else:
        if idx in d2g:
            gs = sorted(d2g[idx], reverse=True)
        elif idx == 0:
            gs = [0]
        else:
            gs = [0]
        for g in gs:
            ga = getAncestors(g, level - 1, d2g)
            return [g] + ga


def myGetAncestors(idx, level, d2g):
    """
    :func find a list of ancestors for the given idx
    :param idx: node idx
    :param level: how many ancestors to find, level 1 return node idx's parent
    :param g2d: governor to dependent mapping
    :param d2g: dependent to governor mapping
    :return: the list of ancestors if level is fulfilled, otherwise None
    """
    if level == 1:
        if idx in d2g:
            return [d2g[idx][0]]
        else:
            return None

    if idx in d2g:
        gs = d2g[idx]
        for g in gs:
            ga = getAncestors(g, level - 1, d2g)
            if ga is None:
                continue
            else:
                return [g] + ga
    return None

