__author__ = 'yandixia'

import numpy as np
import codecs
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plot
import matplotlib
from matplotlib.font_manager import FontProperties


def is_english(word):
    try:
        word.decode('ascii')
    except UnicodeDecodeError:
        return False
    except UnicodeEncodeError:
        return False
    else:
        return True


def word_clustering():
    """
    a script to calculate word cluster.
    :return:
    """
    datafile = '../data/blg250.txt'
    print 'loading w2v file...'
    with open(datafile, 'r') as reader:
        w2vlines = reader.readlines()

    rng = np.random.RandomState(1234)
    perm = rng.permutation(len(w2vlines))
    ratio = 2
    rand_indexes = perm[:len(w2vlines) / ratio]

    data = []
    words = []
    print 'allocating memory...'
    for idx in rand_indexes:
        line = w2vlines[idx]
        words.append(unicode(line.split()[0], encoding='utf8'))
        vector = [np.float32(digit) for digit in line.split()[1:]]
        data.append(vector)

    print 'k-means computing...'
    kmeans = KMeans(init='random', n_clusters=1000, n_init=10)
    kmeans.fit(data)

    print 're-clustering...'
    clusters = {}
    for i, vector in enumerate(data):
        word = words[i]
        label = kmeans.predict(vector)[0]
        if label in clusters:
            clusters[label].append(word)
        else:
            clusters[label] = [word]

    # output word clusters
    print 'output cluster words...'
    outputlines = []
    for cluster, cluster_words in clusters.iteritems():
        outputlines.append(' '.join(cluster_words) + '\n')

    with codecs.open('../exp/word_cluster', 'w', 'utf8') as writer:
        writer.writelines(outputlines)


def create_tsne_input():
    """
    preprocessing.
    :return:
    """
    datafile = '../data/blg250.txt'
    print 'loading w2v file...'
    with open(datafile, 'r') as reader:
        w2vlines = reader.readlines()

    print 'creating dictionary...'
    wd2vec = {}
    for line in w2vlines[1:]:
        try:
            line = unicode(line, encoding='utf8')
            word = line.split()[0]
            vec = '\t'.join(line.split()[1:])
            wd2vec[word] = vec
        except UnicodeDecodeError:
            continue

    print 'reading clusters...'
    cluster_file = '../exp/clusters/children'
    with codecs.open(cluster_file, 'r', 'utf8') as reader:
        clusterlines = reader.readlines()
    words = []
    for clusterline in clusterlines:
        words.extend(clusterline.split())

    vectors = []
    for word in words:
        vectors.append(wd2vec[word])

    print 'output points and labels...'
    with open('../exp/clusters/points', 'w') as writer:
        for vec in vectors:
            writer.write(vec + '\n')

    with codecs.open('../exp/clusters/labels', 'w', 'utf8') as writer:
        for word in words:
            writer.write(word + '\n')


def tsne():
    """
    a scripte to do tsne.
    :return:
    """
    print 'loading...'
    datafile = '../exp/clusters/points'
    data = np.loadtxt(datafile)

    print 'computing t-sne....'
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne = tsne.fit_transform(data)
    np.savetxt('../exp/clusters/2dpoints', X_tsne)


def trim():
    """
    trim the tsne result, in order to make it human readable
    :return:
    """
    trim_threshold = 10
    with codecs.open('../exp/clusters/trimchildren', 'r', 'utf8') as reader:
        clusters = reader.readlines()
    trimwords = []
    for cluster in clusters:
        words = cluster.split()
        eng_count = 0
        eng_words = []
        ch_count = 0
        ch_words = []
        for word in words:
            if is_english(word):
                eng_count += 1
                eng_words.append(word)
            else:
                ch_count += 1
                ch_words.append(word)
        if ch_count > trim_threshold and eng_count > trim_threshold:
            ch_words = ch_words[:trim_threshold]
            eng_words = eng_words[:trim_threshold]
        else:
            if ch_count >= eng_count:
                ch_words = ch_words[:eng_count]
            else:
                eng_words = eng_words[:ch_count]
        trimwords.extend(ch_words + eng_words)

    with codecs.open('../exp/clusters/trimlabels', 'w', 'utf8') as writer:
        for word in trimwords:
            writer.write(word + '\n')

    with codecs.open('../exp/clusters/labels', 'r', 'utf8') as reader:
        labellines = reader.readlines()
    lbl2idx = {}
    for i, line in enumerate(labellines):
        lbl2idx[line.strip()] = i

    trimpoints = []
    with open('../exp/clusters/2dpoints', 'r') as reader:
        pointslines = reader.readlines()

    for word in trimwords:
        trimpoints.append(pointslines[lbl2idx[word]])

    with open('../exp/clusters/trim2dpoints', 'w') as writer:
        writer.writelines(trimpoints)


def visualizedata():
    """
    a script to draw tsne graph.
    :return:
    """
    labelfile = '../exp/clusters/trimlabels'
    with codecs.open(labelfile, 'r', 'utf8') as reader:
        labels = reader.read().split('\n')[:-1]
    X_tsne = np.loadtxt('../exp/clusters/trim2dpoints')

    # prepare plotting
    fontfile = '../exp/clusters/simsun.ttc'
    fontobj = FontProperties(fname=fontfile)
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X = (X_tsne - x_min) / (x_max - x_min + 2)

    fig = plot.figure()
    ax = fig.add_subplot(111)

    for label, x, y in zip(labels, X[:, 0], X[:, 1]):
        ax.text(
            x, y, label, fontproperties=fontobj
        )
    plot.show()


if __name__ == '__main__':
    # create_tsne_input()
    #tsne()
    trim()
    visualizedata()




