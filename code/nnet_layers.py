__author__ = 'xiy1pal'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
import theano.tensor.shared_randomstreams
from theano.tensor.nnet import conv


# different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return (y)


def Tanh(x):
    y = T.tanh(x)
    return (y)


def Iden(x):
    y = x
    return (y)


class HiddenLayer(object):
    """
    Class for HiddenLayer
    """

    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None,
                 use_bias=False):

        self.input = input
        self.activation = activation

        if W is None:
            if activation == ReLU:
                W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)), dtype=theano.config.floatX)
            else:
                W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)),
                                                  size=(n_in, n_out)), dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        if use_bias:
            lin_output = T.dot(input, self.W) + self.b
        else:
            lin_output = T.dot(input, self.W)

        self.output = (lin_output if activation is None else activation(lin_output))

        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out, W=None, b=None):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                value=np.zeros((n_out,), dtype=theano.config.floatX),
                name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                            ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class MLP(object):
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        # the prediction from logistic regression
        self.y_preds = self.logRegressionLayer.y_pred
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape,
                 poolsize,
                 subsample=(1, 1),
                 non_linear="tanh",
                 params=None,
                 image_shape=None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape
        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)
        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)
        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        if params is None:
            # there are "num input feature maps * filter height * filter width"
            # inputs to each hidden unit
            fan_in = np.prod(filter_shape[1:])
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            # initialize weights with random weights
            if self.non_linear == "none" or self.non_linear == "relu":
                self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            else:
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        else:
            filter_index = 0
            bias_index = 1
            self.W = params[filter_index]
            self.b = params[bias_index]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=self.filter_shape,
                               image_shape=self.image_shape, subsample=subsample)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.conv_out = conv_out_tanh
            self.output = T.max(conv_out_tanh, axis=2)
            self.argmax = T.argmax(conv_out_tanh, axis=2)
        elif self.non_linear == "relu":
            conv_out_relu = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.conv_out = conv_out_relu
            self.output = T.max(conv_out_relu, axis=2)
            self.argmax = T.argmax(conv_out_relu, axis=2)
        else:
            self.conv_out = conv_out
            pooled_out = T.max(conv_out, axis=2)
            self.argmax = T.argmax(conv_out, axis=2)
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.params = [self.W, self.b]


class FixedKMaxConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape,
                 poolsize, non_linear="tanh",
                 params=None,
                 image_shape=None):

        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        if params is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            if self.non_linear == "none" or self.non_linear == "relu":
                self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            else:
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        else:
            filter_index = 0
            bias_index = 1
            self.W = params[filter_index]
            self.b = params[bias_index]
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.conv_out = conv_out_tanh
        self.output = T.max(conv_out_tanh, axis=2)
        self.argmax = T.argmax(conv_out_tanh, axis=2)
        self.params = [self.W, self.b]


class KMaxConvPoolLayer(object):
    """Pool Layer of a convolutional network, fixed K max pooling, non-linear function is tanh"""

    def __init__(self,
                 rng,
                 input,
                 filter_shape,
                 poolsize,
                 kmasks,  # shape (batch_size, pos_size, max_mask_len)
                 non_linear="tanh",
                 params=None,
                 image_shape=None):

        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        # parameter initialization
        if params is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            if self.non_linear == "none" or self.non_linear == "relu":
                self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            else:
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        else:
            filter_index = 0
            bias_index = 1
            self.W = params[filter_index]
            self.b = params[bias_index]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        def inner_func(sendid, t4matrix, masks):
            max_ = T.max(t4matrix[sendid, :, masks[sendid], :], axis=1)
            flatten_ = max_.flatten().dimshuffle('x', 0)
            return flatten_

        batch_size = conv_out_tanh.shape[0]
        fm = filter_shape[0]  # feature map

        pad = T.zeros((batch_size, fm, 1, 1), dtype=np.float32)
        cot_pad = T.concatenate([conv_out_tanh, pad], axis=2)

        scan_rs, updates = theano.scan(
            fn=inner_func,
            outputs_info=None,
            sequences=T.arange(batch_size),
            non_sequences=[cot_pad, kmasks]
        )

        conv_pool_out = scan_rs.dimshuffle(0, 2)

        self.output = conv_pool_out

        self.conv_out = conv_out_tanh
        self.argmax = T.argmax(conv_out_tanh, axis=2)
        self.params = [self.W, self.b]


class filterConvPoolLayer(object):
    """Pool Layer of a convolutional network, non-linear function is tanh
        adding hinge score to it.
    """

    def __init__(self,
                 rng,
                 input,
                 filter_shape,
                 poolsize,
                 kmasks,  # shape (batch_size, pos_size, max_mask_len)
                 anti_kmasks,
                 pos_size,
                 filter_divide,
                 non_linear="tanh",
                 params=None,
                 image_shape=None):

        self.input = input
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        # parameter initialization
        if params is None:
            fan_in = np.prod(filter_shape[1:])
            fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) / np.prod(poolsize))
            if self.non_linear == "none" or self.non_linear == "relu":
                self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            else:
                W_bound = np.sqrt(6. / (fan_in + fan_out))
                self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                                  dtype=theano.config.floatX), borrow=True, name="W_conv")
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        else:
            filter_index = 0
            bias_index = 1
            self.W = params[filter_index]
            self.b = params[bias_index]

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=self.filter_shape,
                               image_shape=self.image_shape)
        conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        batch_size = conv_out_tanh.shape[0]
        fm = filter_shape[0]  # feature map

        pad = T.zeros((batch_size, fm, 1, 1), dtype=np.float32)
        cot_pad = T.concatenate([conv_out_tanh, pad], axis=2)

        def inn_func(sendid, prior,t4matrix, masks, anti_masks, pos_idx, filter_divide):
            part_filter = t4matrix[:,filter_divide[pos_idx],:,:]
            cscore = T.mean(part_filter[sendid,:,masks[sendid][pos_idx],:], axis=0)
            elsemax_ = T.max(part_filter[sendid,:,anti_masks[sendid][pos_idx],:], axis=0)
            scores = T.ones_like(cscore) - cscore + elsemax_
            hinge_loss = (scores * (scores > 0)).mean()
            return prior + hinge_loss

        hinge_sum = 0.
        for idx in xrange(pos_size):
            scan_rs, updates = theano.scan(
                fn=inn_func,
                outputs_info=0.,
                sequences=T.arange(batch_size),
                non_sequences=[cot_pad, kmasks, anti_kmasks, idx, filter_divide]
            )
            hinge_sum += scan_rs[-1]

        self.output = T.max(conv_out_tanh, axis=2).flatten(2)

        self.hinge_loss = hinge_sum

        self.conv_out = conv_out_tanh
        self.argmax = T.argmax(conv_out_tanh, axis=2)
        self.params = [self.W, self.b]


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
"""
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
            activation=activation, use_bias=use_bias)

        self.output = _dropout_from_layer(rng, self.output, p=dropout_rate)


class MLPDropout(object):
    """A multilayer perceptron with dropout"""

    def __init__(self, rng, input, layer_sizes, dropout_rate, activation=Tanh, use_bias=True):
        # rectified_linear_activation = lambda x: T.maximum(0.0, x)

        # Set up all the hidden layers
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        next_layer_input = input
        # first_layer = True
        # dropout the input
        next_dropout_layer_input = _dropout_from_layer(rng, input, p=dropout_rate)
        layer_counter = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                                    input=next_dropout_layer_input,
                                                    activation=activation,
                                                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                                                    dropout_rate=dropout_rate)
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = HiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=activation,
                                     # scale the weight matrix W with (1-p)
                                     W=next_dropout_layer.W * (1 - dropout_rate),
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out,
                                     use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            # first_layer = False
            layer_counter += 1

        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W with (1-p)
            W=dropout_output_layer.W * (1 - dropout_rate),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors
        self.y_preds = self.layers[-1].y_pred
        self.p_y_given_x = self.layers[-1].p_y_given_x

        # Grab all the parameters together.
        self.params = [param for layer in self.dropout_layers for param in layer.params]
