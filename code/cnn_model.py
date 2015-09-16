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
        filter_h=3,
        n_epochs=1000,
        batch_size=50,
        feature_maps=100,
        mlphiddensize=100,
        logFile='../exp/logprint',
        logTest='../exp/logTest'
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
    label_c_size = len(clbl_vec)
    label_f_size = len(flbl_vec)
    trainDataSetIndex = 0
    testDataSetIndex = 1
    validDataSetIndex = 2
    sentenceIndex = 0
    clblIndex = 1  # coarse label(clbl) index in the dataset structure
    flblIndex = 2  # fine label(flbl) index
    maskIndex = 3  # a mask matrix representing the length of the sentences, detail in process_data.py

    # train part
    train_c_y = shared_store(datasets[trainDataSetIndex][clblIndex])
    train_f_y = shared_store(datasets[trainDataSetIndex][flblIndex])
    train_x = shared_store(datasets[trainDataSetIndex][sentenceIndex])
    train_mask = datasets[trainDataSetIndex][maskIndex]

    # test part
    gold_test_c_y = datasets[testDataSetIndex][clblIndex]
    gold_test_f_y = datasets[testDataSetIndex][flblIndex]
    test_x = shared_store(datasets[testDataSetIndex][sentenceIndex])
    test_mask = datasets[testDataSetIndex][maskIndex]

    # valid part
    gold_valid_c_y = datasets[validDataSetIndex][clblIndex]
    gold_valid_f_y = datasets[validDataSetIndex][flblIndex]
    valid_x = shared_store(datasets[validDataSetIndex][sentenceIndex])
    valid_mask = datasets[validDataSetIndex][maskIndex]

    train_size = len(datasets[trainDataSetIndex][sentenceIndex])

    w2v = load(w2vFile)
    img_w = w2v.shape[1]  # the dimension of the word embedding
    img_h = len(datasets[trainDataSetIndex][sentenceIndex][0])  # length of each sentence
    max_sent_l = img_h
    filter_w = img_w  # word embedding dimension
    filter_shape = (feature_maps, 1, filter_h, filter_w)
    pool_size = (img_h - filter_h + 1, img_w - filter_w + 1)

    """
    Building model in theano language, less comments here.
    You can refer to Theano web site for more details
    """
    batch_index = T.lvector('hello_index')
    x = T.imatrix('hello_x')
    y_c = T.ivector('hello_y_c')
    y_f = T.ivector('hello_y_f')
    batch_max_l = T.iscalar('hello_max_len')
    test_idx = T.iscalar('hello_test_x')
    single_l = T.iscalar('hello_sent_mask')
    w2v_shared = theano.shared(value=w2v, name='w2v', borrow=True)
    rng = np.random.RandomState(3435)

    input = w2v_shared[x].dimshuffle(0, 'x', 1, 2)

    conv_layer = LeNetConvPoolLayer(
        rng,
        input=input,
        filter_shape=filter_shape,
        poolsize=pool_size,
        image_shape=None
    )

    c_classifier = MLPDropout(
        rng=rng,
        input=conv_layer.output.flatten(2),
        layer_sizes=[feature_maps, mlphiddensize, label_c_size],
        dropout_rate=0.5
    )

    f_classifier = MLPDropout(
        rng=rng,
        input=conv_layer.output.flatten(2),
        layer_sizes=[feature_maps, mlphiddensize, label_f_size],
        dropout_rate=0.5
    )

    # params = [w2v_shared]+conv_layer.params+classifier.params
    params = conv_layer.params + c_classifier.params + f_classifier.params
    cost = c_classifier.negative_log_likelihood(y_c) + f_classifier.negative_log_likelihood(y_f)
    updates = sgd_updates_adadelta(params, cost)

    n_batches = train_x.shape.eval()[0] / batch_size

    # profiling, enable the next two line comments will print out the speed info of each operation
    # profmode = theano.ProfileMode(linker=theano.gof.OpWiseCLinker(), optimizer='fast_run')
    # config.profile = True

    train_model = theano.function(
        inputs=[batch_index, batch_max_l],
        outputs=cost,
        updates=updates,
        givens={
            x: train_x[batch_index][:, :batch_max_l],
            y_c: train_c_y[batch_index],
            y_f: train_f_y[batch_index],
        },
    )

    test_c_model = theano.function(
        inputs=[test_idx, single_l],
        outputs=c_classifier.y_preds,
        givens={
            x: test_x[test_idx][:, :single_l],
        }
    )

    test_f_model = theano.function(
        inputs=[test_idx, single_l],
        outputs=f_classifier.y_preds,
        givens={
            x: test_x[test_idx][:, :single_l],
        }
    )

    valid_c_model = theano.function(
        inputs=[test_idx, single_l],
        outputs=c_classifier.y_preds,
        givens={
            x: test_x[test_idx][:, :single_l],
        }
    )

    valid_f_model = theano.function(
        inputs=[test_idx, single_l],
        outputs=f_classifier.y_preds,
        givens={
            x: test_x[test_idx][:, :single_l],
        }
    )

    """
    Training part
    """
    print 'training....'
    bestep = 0
    bestacc_fc = 0.
    bestacc_f = 0. # best result on valid set
    epoch = 0
    test_acc_c = 0.
    test_acc_f = 0.

    open(logFile, 'w').close()

    # create gold value sequences, required by the eval.py
    with open('../exp/goldcrs', 'w') as writer:
        for lbl in gold_test_c_y:
            writer.write(str(lbl) + '\n')
    with open('../exp/goldfrs', 'w') as writer:
        for lbl in gold_test_f_y:
            writer.write(str(lbl) + '\n')

    # training loop
    while (epoch < n_epochs):
        epoch += 1
        print '************* epoch ' + str(epoch)
        batch_indexes = range(train_size)
        rng.shuffle(batch_indexes)
        for bchidx in xrange(n_batches):
            random_indexes = batch_indexes[bchidx * batch_size:(bchidx + 1) * batch_size]
            max_l = 2 * (filter_h - 1) + train_mask[random_indexes].max()
            train_cost = train_model(random_indexes, max_l)

        valid_c_y_preds = []
        valid_f_y_preds = []
        for idx in xrange(len(datasets[validDataSetIndex][sentenceIndex])):
            valid_c_y_preds.append(valid_c_model(idx, max_sent_l))  # valid_mask[idx]+2*(filter_h-1)))
            valid_f_y_preds.append(valid_f_model(idx, max_sent_l))  # valid_mask[idx]+2*(filter_h-1)))
        valid_acc_c = eval.accuracy(gold_valid_c_y, valid_c_y_preds)
        valid_acc_f = eval.accuracy(gold_valid_f_y, valid_f_y_preds)
        logging(valid_acc_c, valid_acc_f, epoch, logFile)

        if valid_acc_f > bestacc_f or (valid_acc_f == bestacc_f and valid_acc_c > bestacc_fc):
            test_c_y_preds = []
            test_f_y_preds = []
            for idx in xrange(len(datasets[testDataSetIndex][sentenceIndex])):
                test_c_y_preds.append(test_c_model(idx, max_sent_l))  # test_mask[idx]+2*(filter_h-1)))
                test_f_y_preds.append(test_f_model(idx, max_sent_l))  # test_mask[idx]+2*(filter_h-1)))
            test_acc_c = eval.accuracy(gold_test_c_y, test_c_y_preds)
            test_acc_f = eval.accuracy(gold_test_f_y, test_f_y_preds)
            logging(test_acc_c, test_acc_f, epoch, logTest)

        print 'coarse label prediction accuracy of valid set is: ' + str(valid_acc_c)
        print 'fine label prediction accuracy of valid set is: ' + str(valid_acc_f)
        print 'current best fine label prediction accuracy is: ' + str(bestacc_f) + ' at epoch ' + str(
            bestep) + ' with coarse label prediction accurary: ' + str(bestacc_fc)
        print 'coarse label prediction accuracy of test set is: ' + str(test_acc_c)
        print 'fine label prediction accuracy of test set is: ' + str(test_acc_f)

    return bestacc_fc, bestacc_f


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
        writer.write('acc_c: '+str(acc_c)+'; acc_f: '+str(acc_f)+'; epoch: '+str(epoch)+'\n')



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


if __name__ == '__main__':
    w2vFile = '../exp/ch_250.pkl'
    dataFile = '../exp/dataset_ch_qc.pkl'
    labelStructureFile = '../exp/label_struct_ch_qc'
    filter_h=3
    n_epochs=1000
    batch_size=50
    feature_maps=150
    mlphiddensize=60
    logFile='../exp/logprint'
    logTest='../exp/logTest'

    c, f = train_joint_conv_net(
        w2vFile=w2vFile,
        dataFile=dataFile,
        labelStructureFile=labelStructureFile,
        filter_h=filter_h,
        n_epochs=n_epochs,
        batch_size=batch_size,
        feature_maps=feature_maps,
        mlphiddensize=mlphiddensize,
        logFile=logFile,
        logTest=logTest
    )
