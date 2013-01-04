"""
"""
import types
import cPickle
import os
import sys
import time
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tarbm import TARBM


class PrintEverythingMode(theano.Mode):
    def __init__(self):
        def print_eval(i, node, fn):
            print i, node, [input[0] for input in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()],
                                                [print_eval])
        super(PrintEverythingMode, self).__init__(wrap_linker,
                                                  optimizer='fast_run')


class TADBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=[784],
                 hidden_layers_sizes=[400], n_outs=0, sparse=0, delay=0,
                 learning_rate=0.01):
        """This class is made to support a variable number of layers.

        :type numpy_rng: np.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.n_ins = n_ins

        self.numpy_rng = numpy_rng
        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.x_hist = T.matrix('x_history')  # presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
        # of [int] labels
        self.lr = T.dscalar('lr')
        self.delay = delay
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = np.prod(self.n_ins)
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            layer_input = self.x
            layer_input_hist = self.x_hist

            # Construct an RBM that shared weights with this layer
            rbm_layer = TARBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            input_history=layer_input_hist,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            sparse=sparse,
                            delay=delay,
                            lr=self.lr)
            self.rbm_layers.append(rbm_layer)

#        # compute the cost for second phase of training, defined as the
#        # negative log likelihood of the logistic regression (output) layer
#        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
#
#        # compute the gradients with respect to the model parameters
#        # symbolic variable that points to the number of errors made on the
#        # minibatch given by self.x and self.y
#        self.errors = self.logLayer.errors(self.y)

    def save(self, filename):
        if filename is not None:
            f = open(filename, 'wb')
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

    def propup(self, data, layer=0, static=True):
        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history
        lr = T.dscalar()
        rbm = self.rbm_layers[layer]
        # get the cost and the gradient corresponding to one step of CD-15
        [pre_sig, post_sig] = rbm.propup(rbm.input, rbm.input_history)

        #################################
        #     Training the CRBM         #
        #################################
        if static:
            # the purpose of train_crbm is solely to update the CRBM parameters
            fn = theano.function([index],
                                outputs=[pre_sig, post_sig],
                                givens={self.x: data[index]},
                                name='propup_tarbm_static')

        else:
            # the purpose of train_crbm is solely to update the CRBM parameters
            fn = theano.function([index, index_hist, lr],
                                outputs=cost,
                                updates=updates,
                                givens={self.x: train_set_x[index],\
                                self.x_hist: train_set_x[index_hist].reshape((
                                    batch_size,
                                    self.delay * np.prod(self.n_ins))),
                                        self.lr: lr},
                                name='train_atrbm')
        return fn

    def pretraining_functions(self, train_set_x, batch_size, k, layer=0,
                    static=False, all=False, all_with_W=False, binary=False):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''
        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history   
        lr = T.dscalar()

        rbm = self.rbm_layers[layer]
        rbm.binary = binary
        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = rbm.get_cost_updates(k=1, static=static, all=all,
                                             all_with_W=all_with_W)

        #################################
        #     Training the CRBM         #
        #################################
        if static:
            # the purpose of train_crbm is solely to update the CRBM parameters
            fn = theano.function([index, lr],
                                outputs=cost,
                                updates=updates,
                                givens={self.x: train_set_x[index],
                                        self.lr: lr},
                                name='train_atrbm_static')
        else:
            # the purpose of train_crbm is solely to update the CRBM parameters
            fn = theano.function([index, index_hist, lr],
                                outputs=cost,
                                updates=updates,
                                givens={self.x: train_set_x[index],\
                                self.x_hist: train_set_x[index_hist].reshape((
                                    batch_size,
                                    self.delay * np.prod(self.n_ins))),
                                        self.lr: lr},
                                name='train_atrbm')
        return fn

    def pretraining_functions_AE(self, train_set_x, train_set_x_hid,
                            batch_size, delay, corruption=0.5, binary=False):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''
        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history
        lr = T.dscalar()
        pretrain_fns = []
        for rbm in self.rbm_layers:
            rbm.binary = binary
            # get the cost and the gradient corresponding to one step of CD-15
            cost, updates = rbm.get_cost_updates_AE(delay=delay,
                                                    corruption=corruption)

            #################################
            #     Training the CRBM         #
            #################################
            fn = theano.function([index, index_hist, lr],
                                outputs=cost,
                                updates=updates,
                                givens={self.x: train_set_x[index],\
                                    self.x_hist: train_set_x[index_hist]\
                                        .reshape((batch_size,
                                        (delay + 1) * np.prod(self.n_ins))),
                                    #self.x_hist: train_set_x[index_hist],
                                    self.lr: lr},
                                name='train_atrbm_ae', on_unused_input='warn')
            #, mode='DEBUG_MODE')#, mode=PrintEverythingMode())
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[index],
              outputs=self.finetune_cost,
              updates=updates,
              givens={self.x: train_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: train_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={self.x: test_set_x[index * batch_size:
                                            (index + 1) * batch_size],
                         self.y: test_set_y[index * batch_size:
                                            (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={self.x: valid_set_x[index * batch_size:
                                          (index + 1) * batch_size],
                      self.y: valid_set_y[index * batch_size:
                                          (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

    def update_lr(self, error, last_error, lr):
        last_error = np.array(last_error).mean()
        if (error < last_error) and (lr < 1.):
            lr = lr * 1.1
            print 'growing learning rate to ', lr
        elif error >= last_error and (lr > 0.):
            lr = lr * 0.8
            print 'shrinking learning rate to ', lr
        return lr

    def pretrain(self, train_set_x, static_epochs=30, epochs=100,
                 all_epochs=100, lr=0.01, k=1, batch_size=100,
             dbn_filename=None, fig_filename=None, plot_interval=50,
             save_interval=10, seqlen=30, all_with_W=False, binary=False):

        #########################
        # PRETRAINING THE MODEL #
        #########################

        batchdataindex = range(train_set_x.get_value(borrow=True).shape[0])
        n_train_batches = len(batchdataindex) / batch_size
        permindex = np.array(batchdataindex)
        self.numpy_rng.shuffle(permindex)
        ## Pre-train layer-wise
        orig_lr = lr
        for i in xrange(self.n_layers):
            print '... getting the static pretraining functions, layer %d' % i
            pretraining_fn = self.pretraining_functions(
                                    train_set_x=train_set_x,
                                    batch_size=batch_size,
                                    k=k, layer=i, static=True, binary=binary)

            last_err = []
            lr = orig_lr

            for epoch in xrange(static_epochs):
                if plot_interval > 0 and epoch % plot_interval == 0 \
                    and fig_filename != None:
                    fig_name = fig_filename + '_static_epoch_%d' % epoch
                    self.rbm_layers[i].plot_weights(fig_name, self.n_ins,
                                                    static=True)
                # go through the training set
                mean_cost = []
                epoch_time = time.clock()
                for batch_index in xrange(n_train_batches):

                    # indexing is slightly complicated
                    # build a linear index to the starting frames for thisbatch
                    # (i.e. time t) gives a batch_size length array for data
                    data_idx = permindex[batch_index * batch_size:
                                         (batch_index + 1) * batch_size]

                    this_cost = np.array(pretraining_fn(data_idx, lr))

                    #print batch_index, this_cost
                    mean_cost += [this_cost]

                err = np.array(mean_cost).mean(axis=0)
                print ('Static Pre-training layer %i, epoch %d, time %d, cost '
                        % (i, epoch, time.clock() - epoch_time), err)
                if epoch > 0:
                    lr = self.update_lr(err, last_err, lr)
                last_err = [err] + last_err[:2]
                if epoch % save_interval == 0 and dbn_filename != None:
                            self.save(dbn_filename)

#        #if i < self.n_layers:
#        fn = self.propup(train_set_x, layer=i, static=True)
#        for batch_index in xrange(n_train_batches):
#            data_idx = permindex[batch_index * batch_size:(batch_index + 1) \
#                             * batch_size]
#
#            propup = np.array(pretraining_fn(data_idx))
#                    # propup for next layer training
#

        batchdataindex = []
        if not isinstance(seqlen, types.ListType):
            seq = [seqlen for s in range(train_set_x.
                            get_value(borrow=True).shape[0] / seqlen)]
        else:
            seq = seqlen

        last = 0
        for s in seq:
            batchdataindex += range(last + self.delay, last + s)
            last += s
        n_train_batches = len(batchdataindex) / batch_size

        permindex = np.array(batchdataindex)
        self.numpy_rng.shuffle(permindex)
        #permindex = permindex[: 2 * batch_size]
        print len(permindex) % batch_size, len(permindex)
        if (len(permindex) % batch_size) != 0:
            permindex = permindex[:-((len(permindex) % batch_size))]
            print permindex.shape

        for d in range(self.delay):
            print '... getting the AE pretraining functions for delay %d' % d
            xx = T.matrix('xx')
            xx_hist = T.matrix('xx_hist')
            train_set_x_hid = xx
#            outs = self.rbm_layers[0].propup(xx, static=True)
#            fn = theano.function([],outs, givens={xx:train_set_x})
#            [_,train_set_x_hid] = fn()
#            print train_set_x_hid.shape
#            train_set_x_hid = theano.shared(np.asarray(train_set_x_hid,
#                                               dtype=theano.config.floatX))
            # estimate the 'corruption' level
            corruption_level = T.std(xx - xx_hist)
            index = T.lvector()    # index to a [mini]batch
            index_hist = T.lvector()  # index to history
            fn = theano.function([index, index_hist], corruption_level, 
                                 givens={xx:train_set_x[index],
                                xx_hist:train_set_x[index_hist]})
            corrup = fn(permindex, permindex - (d + 1))

            pretraining_fns = \
                        self.pretraining_functions_AE(train_set_x=train_set_x,
                                            train_set_x_hid=train_set_x_hid,
                                            batch_size=batch_size, delay=d,
                                            corruption=corrup, binary=binary)
            ## Pre-train layer-wise
            for i in xrange(self.n_layers):
                lr = orig_lr
                last_err = []
                for epoch in xrange(epochs):
                    if (plot_interval > 0 and epoch % plot_interval == 0
                        and fig_filename != None):
                        fig_name = fig_filename + '_epoch_%d' % epoch
                        self.rbm_layers[i].plot_weights(fig_name, self.n_ins,
                                                        delay=d + 1)
                    # go through the training set
                    mean_cost = []
                    epoch_time = time.clock()
                    #old_weight = self.rbm_layers[i].A.get_value(borrow=False)

                    for batch_index in xrange(n_train_batches):

                        # indexing is slightly complicated
                        # build a linear index to the starting frames for this batch
                        # (i.e. time t) gives a batch_size length array for data
                        data_idx = permindex[batch_index * batch_size:
                                             (batch_index + 1) * batch_size]

                        # now build a linear index to the frames at each delay tap
                        # (i.e. time t-1 to t-delay)
                        # gives a batch_size x delay array of indices for history
                        hist_idx = np.array([data_idx - (d + 1)])

                        hist_idx = np.array([data_idx - n for n
                                             in xrange(1, d + 2)]).T
                        this_cost = np.array(pretraining_fns[i](data_idx,
                                                        hist_idx.ravel(), lr))
                        #weight_diff = old_weight - self.rbm_layers[i].
                        #A.get_value(borrow=True)
                        #print weight_diff.sum(axis=2).sum(axis=1)
                        #print weight_diff
                        #print this_cost.shape, this_cost
                        #print batch_index, this_cost
                        mean_cost += [this_cost]

                    if epoch % save_interval == 0 and dbn_filename != None:
                            self.save(dbn_filename)
                    err = np.array(mean_cost).mean(axis=0)
                    print ('Pre-training layer %i, delay %i, epoch %d, time %d, cost ' % (i,d,
                                        epoch, time.clock() - epoch_time),
                                        err)
                    if epoch > 3:
                        lr = self.update_lr(err, last_err, lr)
                    last_err = [err] + last_err[:2]

#        batchdataindex = []
#        seq = [seqlen for s in range(train_set_x.get_value(borrow=True)
#.shape[0]
#                                      /seqlen)]
#        last = 0
#        for s in seq:
#            batchdataindex += range(last + self.delay, last + s)
#            last += s
#        n_train_batches = len(batchdataindex) / batch_size

        permindex = np.array(batchdataindex)
        self.numpy_rng.shuffle(permindex)
        #permindex = permindex[: 2 * batch_size]
        print len(permindex) % batch_size, len(permindex)
        if (len(permindex) % batch_size) != 0:
            permindex = permindex[:-((len(permindex) % batch_size))]
            print permindex.shape

        print '... getting the AE pretraining functions for ALL'

        ## Pre-train layer-wise
        for i in xrange(self.n_layers):

            pretraining_fn = self.pretraining_functions(
                                        train_set_x=train_set_x,
                                        batch_size=batch_size,
                                        k=k, layer=i, static=False, all=True,
                                        all_with_W=all_with_W, binary=binary)

            lr = orig_lr
            last_err = []
            for epoch in xrange(all_epochs):
                if (plot_interval > 0 and epoch % plot_interval == 0
                    and fig_filename != None):
                    fig_name = fig_filename + '_ALL_epoch_%d' % epoch
                    self.rbm_layers[i].plot_weights(fig_name, self.n_ins,
                                                    delay=d + 1)
                # go through the training set
                mean_cost = []
                epoch_time = time.clock()
                #old_weight = self.rbm_layers[i].A.get_value(borrow=False)

                for batch_index in xrange(n_train_batches):

                    # indexing is slightly complicated
                    # build a linear index to the starting frames for this batch
                    # (i.e. time t) gives a batch_size length array for data
                    data_idx = permindex[batch_index * batch_size:(batch_index + 1) \
                                         * batch_size]

                    # now build a linear index to the frames at each delay tap
                    # (i.e. time t-1 to t-delay)
                    # gives a batch_size x delay array of indices for history
                    hist_idx = np.array([data_idx - (d + 1)])

                    hist_idx = np.array([data_idx - n for n in
                                         xrange(1, d + 2)]).T

                    this_cost = np.array(pretraining_fn(data_idx,
                                                        hist_idx.ravel(), lr))
                    #weight_diff = old_weight - self.rbm_layers[i].A.get_value(borrow=True)
                    #print weight_diff.sum(axis=2).sum(axis=1)
                    #print weight_diff
                    #print this_cost.shape, this_cost
                    #print batch_index, this_cost
                    mean_cost += [this_cost]

                if epoch % save_interval == 0 and dbn_filename != None:
                        self.save(dbn_filename)
                err = np.array(mean_cost).mean(axis=0)
                print ('Pre-training ALL layer %i, epoch %d, time %d, cost ' %
                        (i, epoch, time.clock() - epoch_time), err)

                if epoch > 3:
                    lr = self.update_lr(err, last_err, lr)
                last_err = [err] + last_err[:2]

        if epochs > 0 and plot_interval > 0 and fig_filename != None:
            fig_name = fig_filename + '_finished'
            for d in range(self.delay):
                self.rbm_layers[i].plot_weights(fig_name, self.n_ins,
                                        with_tree=False, delay=d + 1)
