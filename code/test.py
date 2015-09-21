"""
this module is for test only
"""
import codecs


def getVocabFromEnglishDictionary():
    vocablst = set()
    with open('../data/eng_dict_desc.lst', 'r') as reader:
        datalines = reader.readlines()
    for line in datalines:
        if line.strip():
            items = line.split()
            if len(items) < 2:
                continue
            if items[0] == 'Usage':
                continue
            if items[0][0] == '-' or items[0][-1] == '-':
                continue
            if items[0][0] == '\'':
                continue
            if items[0][-1] in [str(i) for i in xrange(10)]:
                vocablst.add(items[0][:-1].lower())
            else:
                vocablst.add(items[0].lower())
    vocablst.add('usage')
    vocablst = sorted(list(vocablst))
    with open('../exp/dict.lst', 'w') as writer:
        for word in vocablst:
            writer.write(word + '\n')


def testUnicode():
    with open('../data/w2v_250.txt', 'r') as reader:
        lines = reader.readlines()
    with open('../exp/vocab_ch_qc.lst', 'r') as reader:
        vlines = reader.readlines()
    w2idx = {}
    count = 0
    for i in xrange(len(vlines)):
        wd = vlines[i].split()[0]
        w2idx[wd] = i
    for line in lines:
        items = line.split()
        wd = items[0]
        if wd in w2idx:
            count += 1
            print wd
    print count


if __name__ == '__main__':
    testUnicode()
