import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


def relu(x):
    return theano.tensor.switch(x<0, 0, x)

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):


        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=relu):

        self.input = input

        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

class ActionLearner(object):
    def __init__(self, rng, input,batch_size, n_filters, n_hidden, n_out):

        #first layer convolutional
        #condense images to 64 x 64

        self.convLayer1 = LeNetConvPoolLayer(
        rng=rng,
        input=input,
        image_shape=(batch_size, 3, 64, 64),
        filter_shape=(n_filters, 3, 5, 5),
        poolsize=(2, 2)
        )

        layer1_input = self.convLayer1.output.flatten(2)

        self.hiddenLayer2 = HiddenLayer(
            rng=rng,
            input=layer1_input,
            n_in=n_filters*30*30,
            n_out=n_hidden,
            activation=relu
        )

        self.hiddenLayer3 = HiddenLayer(
            rng=rng,
            input=self.hiddenLayer2.output,
            n_in=n_hidden,
            n_out=4,
            activation=relu
        )

        self.L2_sqr = (
            (self.convLayer1.W ** 2).sum()
            + (self.hiddenLayer2.W ** 2).sum()
            + (self.hiddenLayer3.W ** 2).sum()
        )

        self.params = self.convLayer1.params + self.hiddenLayer2.params + self.hiddenLayer3.params

        self.input = input

        self.output = self.hiddenLayer3.output

    def single_action_cost(self, y):
        #only get the cost for the nonzero y
        index = numpy.nonzero(y)
        # true_cost = T.scalar()
        true_cost = y[index]
        true_cost = true_cost.copy()
        y = self.output.copy()
        # # y[index] = true_cost
        T.set_subtensor(y[index], true_cost)
        # a = T.vector()
        # b = T.vector()
        # return_cost =(a- b)**2
        # cost_func = theano.function([a,b],return_cost)
        # T.set_subtensor(y[index], self.output[index])
        #returns the euc sq error
        return T.mean((y-self.output)**2)

    def return_function(self,theano_func):
        self.theano_func = theano_func

    def return_action(self,input_variable):
        return self.theano_func(input_variable)
