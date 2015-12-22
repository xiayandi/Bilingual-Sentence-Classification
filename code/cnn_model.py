__author__ = 'xiy1pal'

import theano
import theano.tensor as T
from theano import config
import numpy as np

from nnet_layers import LeNetConvPoolLayer
from nnet_layers import MLPDropout
import nnet_layers

import eval

import cPickle
from collections import OrderedDict
import process_qc
import sys


# theano.config.profile = True


def ReLU(x):
    y = T.maximum(0, x)
    return (y)


def Iden(x):
    y = x
    return (y)


def load(pklfile):
    """
    function: loading pickle file into memory
    :param pklfile:
    :return: the recovered python variable
    """
    pklf = open(pklfile)
    pkldata = cPickle.load(pklf)
    pklf.close()
    return pkldata


def shared_store(data):
    """
    function: convert numpy array variable into theano shared data
    ie. put copy the data into GPU memory, so that GPU can use it
    efficiently.
    :param data: can be array like structure, such as numpy array
    need to be float32 type so that GPU can use it
    :return:
    """
    return theano.shared(data, borrow=True)


def train_joint_conv_net(
        w2vFile,
        dataFile,
        labelStructureFile,
        cfswitch,
        filter_hs,
        n_epochs=1000,
        batch_size=50,
        feature_maps=100,
        mlphiddensize=100,
):
    """
    function: learning and testing sentence level Question Classification Task
            in a joint fashion, ie. adding the loss function of coarse label prediction
            and fine label prediction together.
    :param w2vFile: the path of the word embedding file(pickle file with numpy
            array value, produced by word2vec.py module)
    :param dataFile: the dataset file produced by process_data.py module
    :param labelStructureFile: a file that describes label structure of coarse and fine
            grains. It is produced in produce_data.py in outputlabelstructure()
    "param filter_h: sliding window size.
            *** warning ***
            you cannot just change window size here, if you want to use a different window
            for the experiment. YOU NEED TO RE-PRODUCE A NEW DATASET IN process_data.py
            WITH THE CORRESPONDING WINDOW SIZE.
    :param n_epochs: the number of epochs the training needs to run
    :param batch_size: the size of the mini-batch
    :param feature_maps: how many dimensions you want the abstract sentence
            representation to be
    :param mlphiddensize: the size of the hidden layer in MLP
    :param logFile: the output file of the brief info of each epoch results, basically a
            save for the print out
    :param logTest: keep track of results on test set
    :return: a tuple of best fine grained prediction accuracy and its corresponding
            coarse grained prediction accuracy
    """

    """
    Loading and preparing data
    """
    datasets = load(dataFile)
    clbl_vec, flbl_vec = process_qc.label_structure(labelStructureFile)
    trainDataSetIndex = 0
    testDataSetIndex = 1
    validDataSetIndex = 2
    sentenceIndex = 0
    clblIndex = 1  # coarse label(clbl) index in the dataset structure
    flblIndex = 2  # fine label(flbl) index

    if cfswitch == 'c':
        lblIndex = clblIndex
        label_vec = clbl_vec
    elif cfswitch == 'f':
        lblIndex = flblIndex
        label_vec = flbl_vec
    else:
        print 'wrong arg value in: cfswtich!'
        sys.exit()

    label_size = len(label_vec)

    # train part
    train_y = shared_store(datasets[trainDataSetIndex][lblIndex])
    train_x = shared_store(datasets[trainDataSetIndex][sentenceIndex])

    # test part
    gold_test_y = datasets[testDataSetIndex][lblIndex]
    test_x = shared_store(datasets[testDataSetIndex][sentenceIndex])

    # valid part
    gold_valid_y = datasets[validDataSetIndex][lblIndex]
    valid_x = shared_store(datasets[validDataSetIndex][sentenceIndex])

    w2v = load(w2vFile)
    img_w = w2v.shape[1]  # the dimension of the word embedding
    img_h = len(datasets[trainDataSetIndex][sentenceIndex][0])  # length of each sentence
    filter_w = img_w  # word embedding dimension
    image_shapes = []
    filter_shapes = []
    for i in xrange(len(filter_hs)):
        image_shapes.append((batch_size, 1, img_h, img_w * filter_hs[i]))
        filter_shapes.append((feature_maps, 1, 1, filter_w * filter_hs[i]))

    pool_size = (img_h, 1)

    train_size = len(datasets[trainDataSetIndex][sentenceIndex])
    print 'number of sentences in training set: ' + str(train_size)
    print 'max sentence length: ' + str(len(datasets[trainDataSetIndex][sentenceIndex][0]))
    print 'train data shape: ' + str(datasets[trainDataSetIndex][sentenceIndex].shape)
    print 'word embedding dim: ' + str(w2v.shape[1])

    """
    Building model in theano language, less comments here.
    You can refer to Theano web site for more details
    """
    batch_index = T.lvector('hello_batch_index')
    x = T.itensor3('hello_x')
    y = T.ivector('hello_y')
    w2v_shared = theano.shared(value=w2v, name='w2v', borrow=True)
    rng = np.random.RandomState(3435)

    conv_layer_outputs = []
    conv_layers = []
    for i in xrange(len(filter_hs)):
        input = w2v_shared[x.flatten()].reshape(
            (x.shape[0], 1, x.shape[1], x.shape[2] * img_w)
        )[:, :, :, 0:filter_hs[i] * img_w]

        conv_layer = LeNetConvPoolLayer(
            rng,
            input=input,
            filter_shape=filter_shapes[i],
            poolsize=pool_size,
            image_shape=image_shapes[i],
            non_linear="relu"
        )

        conv_layers.append(conv_layer)
        conv_layer_outputs.append(conv_layer.output.flatten(2))

    mlp_input = T.concatenate(conv_layer_outputs, 1)

    classifier = MLPDropout(
        rng=rng,
        input=mlp_input,
        layer_sizes=[feature_maps * len(filter_hs), label_size],
        dropout_rate=0.5,
        activation=Iden
    )

    params = []
    for conv_layer in conv_layers:
        params += conv_layer.params
    params += classifier.params

    cost = classifier.negative_log_likelihood(y)
    updates = sgd_updates_adadelta(params, cost)

    n_batches = train_x.shape.eval()[0] / batch_size

    train_model = theano.function(
        inputs=[batch_index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[batch_index],
            y: train_y[batch_index],
        },
    )

    """
    Building test model
    """
    test_conv_layer_outputs = []
    for i, conv_layer in enumerate(conv_layers):
        test_input = w2v_shared[x.flatten()].reshape(
            (x.shape[0], 1, x.shape[1], x.shape[2] * img_w)
        )[:, :, :, 0:filter_hs[i] * img_w]
        test_conv_layer_outputs.append(
            conv_layer.conv_layer_output(
                test_input,
                (test_x.shape.eval()[0], 1, img_h, img_w * filter_hs[i])
            ).flatten(2)
        )
    test_prediction = classifier.predict(T.concatenate(test_conv_layer_outputs, 1))

    # test on test set
    test_model = theano.function(
        inputs=[],
        outputs=test_prediction,
        givens={
            x: test_x,
        }
    )

    # test on valid set
    valid_model = theano.function(
        inputs=[],
        outputs=test_prediction,
        givens={
            x: valid_x,
        }
    )

    """
    Training part
    """
    print 'training....'
    best_valid_ep = 0
    best_valid_acc = 0.
    best_test_ep = 0
    best_test_acc = 0.
    final_acc = 0.
    epoch = 0

    # create gold value sequences, required by the eval.py
    with open('../exp/goldrs', 'w') as writer:
        for lbl in gold_test_y:
            writer.write(str(lbl) + '\n')

    # training loop
    while (epoch < n_epochs):
        epoch += 1
        print '************* epoch ' + str(epoch)
        batch_indexes = range(train_size)
        rng.shuffle(batch_indexes)
        for bchidx in xrange(n_batches):
            random_indexes = batch_indexes[bchidx * batch_size:(bchidx + 1) * batch_size]
            train_cost = train_model(random_indexes)

        test_y_preds = test_model()
        valid_y_preds = valid_model()
        test_acc = eval.accuracy(gold_test_y, test_y_preds)
        valid_acc = eval.accuracy(gold_valid_y, valid_y_preds)
        #test_acc = eval.fscore(gold_test_y, test_y_preds)
        #valid_acc = eval.fscore(gold_valid_y, valid_y_preds)
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_valid_ep = epoch
            if final_acc < test_acc:
                final_acc = test_acc
                with open('../exp/predictions', 'w') as writer:
                    for lblidx in test_y_preds:
                        writer.write(str(lblidx) + '\n')
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_ep = epoch
            # output predictions

        print 'test accuracy is: ' + str(test_acc)
        print 'valid accuracy is: ' + str(valid_acc)
        print 'current best valid prediction accuracy is: ' + str(best_valid_acc) + ' at epoch ' + str(best_valid_ep)
        print 'current best test prediction accuracy is: ' + str(best_test_acc) + ' at epoch ' + str(best_test_ep)

    return final_acc


def logging(acc_c, acc_f, epoch, logfile):
    """
    function: append new test result in the end of logfile
    :param acc_c: coarse label prediction accuracy
    :param acc_f: fine label prediction accuracy
    :param epoch: the epoch that this result is reached
    :param logfile: the log file
    :return: None
    """
    with open(logfile, 'a') as writer:
        writer.write('acc_c: ' + str(acc_c) + '; acc_f: ' + str(acc_f) + '; epoch: ' + str(epoch) + '\n')


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


###############################################################################################
###############################################################################################
###############################################################################################
#####                              Experiment Section                                     #####
###############################################################################################
###############################################################################################
###############################################################################################


def general_rundown():
    w2vFile = '../exp/blg250.pkl'
    dataFile = '../exp/dataset_bi.pkl'
    labelStructureFile = '../exp/label_struct_bi'
    cfswitch = 'c'

    n_epochs = 20

    ###### QC parameter: universal/collapse + lexicon + phrase #######
    filter_hs = [3, 4, 5]
    batch_size = 50
    feature_maps = 100

    # process_qc.datasetConstructRundown(10, 0)

    acc = train_joint_conv_net(
        w2vFile=w2vFile,
        dataFile=dataFile,
        labelStructureFile=labelStructureFile,
        cfswitch=cfswitch,
        filter_hs=filter_hs,
        n_epochs=n_epochs,
        batch_size=batch_size,
        feature_maps=feature_maps,
    )


def rundown_qc():
    w2vFile = '../exp/blg250.pkl'
    dataFile = '../exp/dataset_bi.pkl'
    labelStructureFile = '../exp/label_struct_bi'
    cfswitch = 'c'

    n_epochs = 100

    ###### QC parameter: universal/collapse + lexicon + phrase #######
    filter_hs = [1, 3]  # , 4]#, 5]
    batch_size = 120
    feature_maps = 110
    mlphiddensize = 20

    """
    ###### QC parameter: universal/collapse 74.27#######
    filter_hs = [3, 4, 5]
    batch_size = 180
    feature_maps = 90
    mlphiddensize = 20

    """

    # process_qc.datasetConstructRundown(10, 0)

    acc = train_joint_conv_net(
        w2vFile=w2vFile,
        dataFile=dataFile,
        labelStructureFile=labelStructureFile,
        cfswitch=cfswitch,
        filter_hs=filter_hs,
        n_epochs=n_epochs,
        batch_size=batch_size,
        feature_maps=feature_maps,
        mlphiddensize=mlphiddensize,
    )


def rundown_mr():
    w2vFile = '../exp/blg250.pkl'
    dataFile = '../exp/dataset_bi.pkl'
    labelStructureFile = '../exp/label_struct_bi'
    cfswitch = 'c'

    n_epochs = 100

    ###### QC parameter #######
    filter_hs = [1, 3]  # , 4]#, 5]
    batch_size = 120
    feature_maps = 110
    mlphiddensize = 20

    ###### movie review parameter ######
    # filter_hs = [1, 3, 4, 5]
    #batch_size = 170
    #feature_maps = 100  # 150
    #mlphiddensize = 60

    # process_qc.datasetConstructRundown(10, 0)

    acc = train_joint_conv_net(
        w2vFile=w2vFile,
        dataFile=dataFile,
        labelStructureFile=labelStructureFile,
        cfswitch=cfswitch,
        filter_hs=filter_hs,
        n_epochs=n_epochs,
        batch_size=batch_size,
        feature_maps=feature_maps,
        mlphiddensize=mlphiddensize,
    )


def exprun():
    w2vFile = '../exp/blg250.pkl'
    dataFile = '../exp/dataset_bi.pkl'
    labelStructureFile = '../exp/label_struct_bi'
    cfswitch = 'c'
    filter_hs = [1, 3]  # , 4, 5]
    n_epochs = 20
    batch_size = 170
    feature_maps = 100  # 150
    mlphiddensize = 60
    logFile = '../exp/logprint'
    logTest = '../exp/logTest'

    ch_pps = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    accs = []

    for i, pp in enumerate(ch_pps):
        process_qc.datasetConstructRundown(10, pp)
        acc = train_joint_conv_net(
            w2vFile=w2vFile,
            dataFile=dataFile,
            labelStructureFile=labelStructureFile,
            cfswitch=cfswitch,
            filter_hs=filter_hs,
            n_epochs=n_epochs,
            batch_size=batch_size,
            feature_maps=feature_maps,
            mlphiddensize=mlphiddensize,
        )
        accs.append(acc)
    print accs


def structureSelection(filter_hs, logFile):
    w2vFile = '../exp/blg250.pkl'
    dataFile = '../exp/dataset_bi.pkl'
    labelStructureFile = '../exp/label_struct_bi'
    cfswitch = 'f'
    #filter_hs = [1, 3]  # [1, 3] # , 4]#, 5]
    n_epochs = 10
    batch_sizes = [100, 120, 140, 160, 170, 180, 200, 220, 240]
    feature_mapss = [50, 70, 90, 100, 110, 130, 150]
    #logFile = '../exp/selectionlog'
    open(logFile, 'w').close()
    best_acc = 0.0
    best_batch_size = 0
    best_fm = 0

    for batch_size in batch_sizes:
        for feature_maps in feature_mapss:
            acc = train_joint_conv_net(
                w2vFile=w2vFile,
                dataFile=dataFile,
                labelStructureFile=labelStructureFile,
                cfswitch=cfswitch,
                filter_hs=filter_hs,
                n_epochs=n_epochs,
                batch_size=batch_size,
                feature_maps=feature_maps,
            )
            if acc > best_acc:
                best_acc = acc
                best_batch_size = batch_size
                best_fm = feature_maps
                with open(logFile, 'a') as writer:
                    writer.write(
                        'best acc:' + str(best_acc) + ' best_btch:' + str(best_batch_size) + ' best_fm:' + str(
                            best_fm) + '\n')


def script(config_, filter_hss):
    for i, filter_hs in enumerate(filter_hss):
        logfile = config_.logprefix+str(i)
        structureSelection(filter_hs, logfile)


if __name__ == '__main__':
    print 'cnn model'
