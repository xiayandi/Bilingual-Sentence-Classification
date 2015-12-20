__author__ = 'xiy1pal'
from process_qc import label_structure


_pathBase_ = '/home/yandixia/workspace/qa/'


def accuracy(goldlblseq, predlblseq):
    assert len(goldlblseq) == len(predlblseq)
    acc = 0.
    crct_count = 0.
    total_count = len(goldlblseq)
    for i in xrange(len(goldlblseq)):
        if goldlblseq[i] == predlblseq[i]:
            crct_count += 1.
    acc = crct_count/total_count
    print 'correct: '+str(crct_count)+' total: '+str(total_count)
    return acc


def fscore(goldlblseq, predlblseq):
    assert len(goldlblseq) == len(predlblseq)
    truecount = 0.
    predcount = 0.
    hitcount = 0.
    for i in xrange(len(goldlblseq)):
        if goldlblseq[i] == 1:
            truecount += 1.
        if predlblseq[i] == 1:
            predcount += 1
        if goldlblseq[i] == 1 and predlblseq[i] == 1:
            hitcount += 1
    recall = hitcount/truecount
    prec = hitcount/predcount
    fscore = 2.*recall*prec/(recall+prec)
    print 'fscore: '+str(fscore)+' prec: '+str(prec)+' recall: '+str(recall)
    return fscore


def outputErrorInstances():
    bestfFile = _pathBase_+'exp/bestjointcnnfrs'
    goldfrsFile = _pathBase_+'exp/goldfrs'
    rcnstrct_sents_file = _pathBase_+'exp/reconstructed_sentences'
    maxplrsFile = _pathBase_+'exp/feature_map_max'

    error_output_file = _pathBase_+'exp/error_max_pool_info'

    with open(bestfFile, 'r') as reader:
        predlblseq = [int(line) for line in reader.readlines()]
    with open(goldfrsFile, 'r') as reader:
        goldlblseq = [int(line) for line in reader.readlines()]
    with open(rcnstrct_sents_file, 'r') as reader:
        sentseq = [line.rstrip() for line in reader.readlines()]
    with open(maxplrsFile, 'r') as reader:
        maxplseq = [[int(strint) for strint in line.split()] for line in reader.readlines()]
    assert len(goldlblseq) == len(predlblseq) == len(sentseq) == len(maxplseq)

    c_vec, f_vec = label_structure('label_structure_new')

    error_instances = []
    maxplinfolines = []
    count = 0
    for i in xrange(len(goldlblseq)):
        if goldlblseq[i] != predlblseq[i]:
            count += 1
            words = sentseq[i].split()
            error_instances.append(f_vec[predlblseq[i]]+'\t'+f_vec[goldlblseq[i]]+'\t'+sentseq[i])
            maxplinfo = ''
            for j in xrange(len(maxplseq[i])):
                argmax = maxplseq[i][j]
                threegram = str(j)+':'+words[argmax]+'/'+words[argmax+1]+'/'+words[argmax+2]
                maxplinfo += threegram+' '
            maxplinfolines.append(maxplinfo.rstrip())

    assert len(error_instances) == len(maxplinfolines)
    print count

    with open(error_output_file, 'w') as writer:
        for i in xrange(len(error_instances)):
            writer.write(error_instances[i]+'\n')
            writer.write(maxplinfolines[i]+'\n')


def outputAmplifierArgmaxpool():
    sents_file = _pathBase_+'exp/caetest'
    maxplrsFile = _pathBase_+'exp/feature_map_max'

    error_output_file = _pathBase_+'exp/error_max_pool_info'

    with open(sents_file, 'r') as reader:
        sentseq = [line.rstrip() for line in reader.readlines()]
    with open(maxplrsFile, 'r') as reader:
        maxplseq = [[int(strint) for strint in line.split()] for line in reader.readlines()]
    assert len(sentseq) == len(maxplseq)

    error_instances = []
    maxplinfolines = []
    for i in xrange(len(sentseq)):
        words = sentseq[i].split()
        error_instances.append(sentseq[i])
        maxplinfo = ''
        for j in xrange(len(maxplseq[i])):
            argmax = maxplseq[i][j]
            argword = str(j)+':'+words[argmax]
            maxplinfo += argword+' '
        maxplinfolines.append(maxplinfo.rstrip())

    assert len(error_instances) == len(maxplinfolines)

    with open(error_output_file, 'w') as writer:
        for i in xrange(len(error_instances)):
            writer.write(error_instances[i]+'\n')
            writer.write(maxplinfolines[i]+'\n')


def outputCAEargmaxpool():
    rcnstrct_sents_file = _pathBase_+'exp/reconstructed_sentences'
    maxplrsFile = _pathBase_+'exp/feature_map_max'

    error_output_file = _pathBase_+'exp/error_max_pool_info'

    with open(rcnstrct_sents_file, 'r') as reader:
        sentseq = [line.rstrip() for line in reader.readlines()]
    with open(maxplrsFile, 'r') as reader:
        maxplseq = [[int(strint) for strint in line.split()] for line in reader.readlines()]
    assert len(sentseq) == len(maxplseq)

    error_instances = []
    maxplinfolines = []
    for i in xrange(len(sentseq)):
        words = sentseq[i].split()
        error_instances.append(sentseq[i])
        maxplinfo = ''
        for j in xrange(len(maxplseq[i])):
            argmax = maxplseq[i][j]
            threegram = str(j)+':'+words[argmax]+'/'+words[argmax+1]+'/'+words[argmax+2]
            maxplinfo += threegram+' '
        maxplinfolines.append(maxplinfo.rstrip())

    assert len(error_instances) == len(maxplinfolines)

    with open(error_output_file, 'w') as writer:
        for i in xrange(len(error_instances)):
            writer.write(error_instances[i]+'\n')
            writer.write(maxplinfolines[i]+'\n')

def lbl2index():
    c_vec, f_vec = label_structure('./label_structure_new')
    c2idx = {}
    f2idx = {}
    for i in xrange(len(c_vec)):
        c2idx[c_vec[i]] = i
    for i in xrange(len(f_vec)):
        f2idx[f_vec[i]] = i
    return c2idx, f2idx


def confusionMatrix():
    switch = 'c'
    with open(_pathBase_+'exp/bestcnn'+switch+'rs', 'r') as reader:
        rslines = reader.readlines()
    with open(_pathBase_+'exp/goldrs', 'r') as reader:
        truelines = reader.readlines()
    with open(_pathBase_+'data/boschtest_new', 'r') as reader:
        testlines = reader.readlines()
    assert len(truelines) == len(rslines) == len(testlines)
    c2idx, f2idx = lbl2index()
    predslines = [int(line) for line in rslines]

    testcseq = [c2idx[line.split()[0].split(':')[0]] for line in testlines]
    testfseq = [f2idx[line.split()[0]] for line in testlines]

    c_vec, f_vec = label_structure('./label_structure_new')
    if switch == 'c':
        lbl_vec = c_vec
        goldlines = testcseq
    else:
        lbl_vec = f_vec
        goldlines = testfseq

    cm = [[0 for i in xrange(len(lbl_vec))] for j in xrange(len(lbl_vec))]

    error_instances = []
    for i in xrange(len(goldlines)):
        glbl = goldlines[i]
        plbl = predslines[i]
        print plbl
        print glbl
        cm[plbl][glbl] += 1
        if plbl != glbl:
            error_instances.append(lbl_vec[plbl]+'\t'+testlines[i])

    tmpline = ''
    for i in xrange(len(lbl_vec)):
        tmpline += str(i)+': '+lbl_vec[i] + '\t'
        if i%5 == 0:
            print tmpline
            tmpline = ''
    print tmpline

    tmpline = '    ['
    for i in xrange(len(cm)):
        if i < 10:
            tmpline += ' ' + str(i)+', '
        else:
            tmpline += str(i)+', '
    print tmpline.rstrip(', ')+']'

    for i in xrange(len(cm)):
        row = cm[i]
        tmpline = '['
        for num in row:
            if num >= 10:
                tmpline += str(num)+', '
            else:
                tmpline += ' '+str(num)+', '
        tmpline = tmpline.rstrip(', ')+']'
        if i < 10:
            print '[ '+str(i)+']'+tmpline
        else:
            print '['+str(i)+']'+tmpline
    for line in error_instances:
        print line.strip()

if __name__ == '__main__':
    #confusionMatrix()
    outputErrorInstances()
    #outputCAEargmaxpool()
    #outputAmplifierArgmaxpool()





