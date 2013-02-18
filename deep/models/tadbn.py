""" Theano TADBN Implementation
@author Chris Hausler - Freie Universitaet Berlin

Modified from the Deep Learning Tutorials
http://deeplearning.net/tutorial/code/DBN.py
"""
import types
import cPickle
import time
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from tarbm import TARBM


class PrintEverythingMode(theano.Mode):
    """Print everything from a Theano variable

    use this theano mode to print many details for debugging
    """
    def __init__(self):
        def print_eval(i, node, fn):
            print i, node, [input_[0] for input_ in fn.inputs],
            fn()
            print [output[0] for output in fn.outputs]
        wrap_linker = theano.gof.WrapLinkerMany([theano.gof.OpWiseCLinker()],
                                                [print_eval])
        super(PrintEverythingMode, self).__init__(wrap_linker,
                                                  optimizer='fast_run')


class TADBN(object):
    """Deep Belief Network for TARBMs

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=[784],
                 hidden_layers_sizes=[400], sparse=0, delay=0,
                 learning_rate=0.01):
        """This class is made to support a variable number of layers.

        Args:
            numpy_rng: a np.random.RandomState, numpy random number generator
                        used to draw initial weights

            theano_rng: theano.tensor.shared_randomstreams.RandomStreams.
                        if None is given one is generated based on a seed
                        drawn from `rng`

            n_ins: list of ints, dimension of the input to the DBN

            n_layers_sizes: list of ints for the size of each hidden layer
                        in the DBN
            sparse: float, the target sparsity value of hidden unit activations
                    ignored if 0
            delay: int, the temporal dependance of the model in frames
            learning_rate: float, the initial learning rate of the model
        """

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
        self.x_hist = T.matrix('x_hist')  # presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector
        self.lr = T.dscalar('lr')  # learning rate
        self.delay = delay  # model delay

        for i in xrange(self.n_layers):

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

            #TODO so far only works for one layer!!
            layer_input = self.x
            layer_input_hist = self.x_hist

            # Construct an RBM that shared weights with this layer
            rbm_layer = TARBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input_=layer_input,
                            input_history=layer_input_hist,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            sparse=sparse,
                            delay=delay,
                            lr=self.lr)
            self.rbm_layers.append(rbm_layer)

    def save(self, filename):
        """Save the TADBN with cPickle

        Note: if you create and save the model on a machine with a GPU, you
        will need a GPU on the machine where you want to open the saved model

        Args:
            filename: The name of the file to save to
        """
        if filename is not None:
            f = open(filename, 'wb')
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

    def propup(self, data, layer=0, static=False):
        """
        propogate the activity through layer 0 to the hidden layer and return
        an array of [2, samples, dimensions]
        where the first 2 dimensions are
        [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]
        so far only works for the first rbm layer
        """
        if not isinstance(data,
                          theano.tensor.sharedvar.TensorSharedVariable):
            data = theano.shared(data)

        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history
        rbm = self.rbm_layers[layer]
        # get the cost and the gradient corresponding to one step of CD-15
        [pre_sig, post_sig] = rbm.propup(static)

        #################################
        #     Training the CRBM         #
        #################################
        if static:
            # the purpose of train_crbm is solely to update the CRBM parameters
            fn = theano.function([],
                                outputs=[pre_sig, post_sig],
                                givens={self.x: data},
                                name='propup_tarbm_static')
            return np.array(fn())

        else:
            # indexing is slightly complicated
            # build a linear index to the starting frames for this batch
            # (i.e. time t) gives a batch_size length array for data
            data_idx = np.arange(self.delay, 
                                 data.get_value(borrow=True).shape[0])

            # now build a linear index to the frames at each delay tap
            # (i.e. time t-1 to t-delay)
            # gives a batch_size x delay array of indices for history
            hist_idx = np.array([data_idx - n for n in
                                 xrange(1, self.delay + 1)]).T

            # the purpose of train_crbm is solely to update the CRBM parameters
            fn = theano.function([index, index_hist],
                                outputs=[pre_sig, post_sig],
                                givens={self.x: data[index],\
                                self.x_hist: data[index_hist].reshape((
                                    len(data_idx),
                                    self.delay * np.prod(self.n_ins)))},
                                name='train_tarbm')

            return np.array(fn(data_idx, hist_idx.ravel()))




    def pretraining_functions(self, train_set_x, batch_size, k, layer=0,
                    static=False, with_W=False, binary=False):
        """Creates functions for doing CD

        Generates a function for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        Args:
            train_set_x: Shared var. that contains all datapoints used
                                for training the RBM
            batch_size: int, the size of each minibatch
            k: number of Gibbs steps to do in CD-k / PCD-k
            layer: which layer of the dbn to generate functions for
            static: if True, ignore all temporal components
            with_W: Whether or not to include the W in update
            binary: if true, make visible layer binary
        Returns:
            CD function

        """
        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history
        lr = T.dscalar()

        rbm = self.rbm_layers[layer]
        rbm.binary = binary
        # get the cost and the gradient corresponding to one step of CD-15
        cost, updates = rbm.get_cost_updates(k=k, static=static, with_W=with_W)

        #################################
        #     Training the RBM         #
        #################################
        if static:
            # updates only on non-temporal components
            fn = theano.function([index, lr],
                                outputs=cost,
                                updates=updates,
                                givens={self.x: train_set_x[index],
                                        self.lr: lr},
                                name='train_tarbm_static')
        else:
            # updates including temporal components
            fn = theano.function([index, index_hist, lr],
                                outputs=cost,
                                updates=updates,
                                givens={self.x: train_set_x[index],\
                                self.x_hist: train_set_x[index_hist].reshape((
                                    batch_size,
                                    self.delay * np.prod(self.n_ins))),
                                        self.lr: lr},
                                name='train_tarbm')
        return fn

    def pretraining_functions_AE(self, train_set_x, batch_size, delay,
                                 layer=0, corruption=0.5, binary=False):
        """ Generates a functions for performing temporal autoencoding
        on one temporal weight set (one delay). The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        Args:
            train_set_x: Shared var. that contains all datapoints used
                                for training the RBM
            batch_size: int, the size of each minibatch
            delay: which temporal delay to train
            layer: which layer of the dbn to generate functions for
            corruption: estimate of the 'corruption' between current datapoints
                and historical ones
            binary: if true, make visible layer binary
        Returns:
            CD function
        """
        # allocate symbolic variables for the data
        index = T.lvector()    # index to a [mini]batch
        index_hist = T.lvector()  # index to history
        lr = T.dscalar()
        rbm = self.rbm_layers[layer]
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
                            name='train_tarbm_ae', on_unused_input='warn')
        #, mode='DEBUG_MODE')#, mode=PrintEverythingMode())

        return fn

    def update_lr(self, error, last_error, lr):
        """ Grow or shrink learning rate based on performance
        Grow LR if error is decreasing
        Shrink LR if error is increasing

        Args:
            error: the current error value
            last_error: a list of the last error values
            lr: the current learning rate
        Returns:
            new learning rate
        """
        last_error = np.array(last_error).mean()
        if (error < last_error) and (lr < 1.):
            lr = lr * 1.1
            print 'growing learning rate to ', lr
        elif error >= last_error and (lr > 0.):
            lr = lr * 0.8
            print 'shrinking learning rate to ', lr
        return lr

    def _do_static_training(self, layer,
                            train_set_x, epochs, lr, k,
                            batch_size, binary, dbn_filename, fig_filename,
                            save_interval, plot_interval):

        print '... getting the static pretraining functions, layer %d' % layer
        pretraining_fn = self.pretraining_functions(
                                train_set_x=train_set_x,
                                batch_size=batch_size,
                                k=k, layer=layer, static=True, binary=binary)

        batchdataindex = range(train_set_x.get_value(borrow=True).shape[0])
        n_train_batches = len(batchdataindex) / batch_size
        permindex = np.array(batchdataindex)
        self.numpy_rng.shuffle(permindex)

        last_err = []
        for epoch in xrange(epochs):
            if plot_interval > 0 and epoch % plot_interval == 0 \
                and fig_filename != None:
                fig_name = fig_filename + '_static_epoch_%d' % epoch
                self.rbm_layers[layer].plot_weights(fig_name, self.n_ins,
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
                    % (layer, epoch, time.clock() - epoch_time), err)
            if epoch > 0:
                lr = self.update_lr(err, last_err, lr)
            last_err = [err] + last_err[:5]
            if epoch % save_interval == 0 and dbn_filename != None:
                        self.save(dbn_filename)

    def _get_temporal_batches(self, seqlen, data_len, batch_size):
        batchdataindex = []
        if not isinstance(seqlen, types.ListType):
            seq = [seqlen for s in range(data_len / seqlen)]
        else:
            seq = seqlen

        last = 0
        for s in seq:
            batchdataindex += range(last + self.delay, last + s)
            last += s
        n_train_batches = len(batchdataindex) / batch_size
        permindex = np.array(batchdataindex)
        self.numpy_rng.shuffle(permindex)
        print len(permindex) % batch_size, len(permindex)
        if (len(permindex) % batch_size) != 0:
            permindex = permindex[:-((len(permindex) % batch_size))]
            print permindex.shape

        return permindex, n_train_batches

    def _do_ae_training(self, permindex, n_train_batches, layer,
                            train_set_x, epochs, lr,
                            batch_size, binary, dbn_filename, fig_filename,
                            save_interval, plot_interval):
        for d in range(self.delay):
            print '... getting the AE pretraining functions for delay %d' % d
            xx = T.matrix('xx')
            xx_hist = T.matrix('xx_hist')

            # estimate the 'corruption' level
            corruption_level = T.std(xx - xx_hist)
            index = T.lvector()    # index to a [mini]batch
            index_hist = T.lvector()  # index to history
            fn = theano.function([index, index_hist], corruption_level,
                                 givens={xx: train_set_x[index],
                                xx_hist: train_set_x[index_hist]})
            corrup = fn(permindex, permindex - (d + 1))

            ## Pre-train layer-wise
            pretraining_fn = \
                    self.pretraining_functions_AE(train_set_x=train_set_x,
                                        batch_size=batch_size, delay=d,
                                        corruption=corrup, binary=binary,
                                        layer=layer)
            last_err = []
            for epoch in xrange(epochs):
                if (plot_interval > 0 and epoch % plot_interval == 0
                    and fig_filename != None):
                    fig_name = fig_filename + '_epoch_%d' % epoch
                    self.rbm_layers[layer].plot_weights(fig_name, self.n_ins,
                                                    delay=d + 1)
                # go through the training set
                mean_cost = []
                epoch_time = time.clock()

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
                    this_cost = np.array(pretraining_fn(data_idx,
                                                    hist_idx.ravel(), lr))
                    mean_cost += [this_cost]

                if epoch % save_interval == 0 and dbn_filename != None:
                        self.save(dbn_filename)
                err = np.array(mean_cost).mean(axis=0)
                print ('Pre-training layer %i, delay %i, epoch %d, time %d,'
                       ' cost ' % (layer, d, epoch, time.clock() - epoch_time),
                                    err)
                if epoch > 3:
                    lr = self.update_lr(err, last_err, lr)
                last_err = [err] + last_err[:5]

    def _do_all_training(self, permindex, n_train_batches, layer,
                            train_set_x, epochs, lr, k,
                            batch_size, binary, with_W,
                            dbn_filename, fig_filename,
                            save_interval, plot_interval):

        print '... getting the pretraining functions for ALL'
        pretraining_fn = self.pretraining_functions(
                                        train_set_x=train_set_x,
                                        batch_size=batch_size,
                                        k=k, layer=layer, static=False,
                                        with_W=with_W, binary=binary)

        last_err = []
        for epoch in xrange(epochs):
            if (plot_interval > 0 and epoch % plot_interval == 0
                and fig_filename != None):
                fig_name = fig_filename + '_ALL_epoch_%d' % epoch
                self.rbm_layers[layer].plot_weights(fig_name, self.n_ins,
                                                    delay=self.delay)
            # go through the training set
            mean_cost = []
            epoch_time = time.clock()
            #old_weight = self.rbm_layers[i].A.get_value(borrow=False)

            for batch_index in xrange(n_train_batches):

                # indexing is slightly complicated
                # build a linear index to the starting frames for this batch
                # (i.e. time t) gives a batch_size length array for data
                data_idx = permindex[batch_index * batch_size:(batch_index + 1)
                                     * batch_size]

                # now build a linear index to the frames at each delay tap
                # (i.e. time t-1 to t-delay)
                # gives a batch_size x delay array of indices for history
                hist_idx = np.array([data_idx - n for n in
                                     xrange(1, self.delay + 1)]).T

                this_cost = np.array(pretraining_fn(data_idx,
                                                    hist_idx.ravel(), lr))
                mean_cost += [this_cost]

            if epoch % save_interval == 0 and dbn_filename != None:
                    self.save(dbn_filename)
            err = np.array(mean_cost).mean(axis=0)
            print ('Pre-training ALL layer %i, epoch %d, time %d, cost ' %
                    (layer, epoch, time.clock() - epoch_time), err)

            if epoch > 3:
                lr = self.update_lr(err, last_err, lr)
            last_err = [err] + last_err[:5]

    def pretrain(self, train_set_x, static_epochs=30, ae_epochs=100,
                 all_epochs=100, lr=0.01, k=1, batch_size=100,
                 seqlen=30, with_W=False, binary=False,
                 dbn_filename=None, fig_filename=None,
                 save_interval=10, plot_interval=50):
        """Do unsupervised training on the model

        There are 3 training steps.
            1) static training on the data. Does not take into account
                temporal connections
            2) autoencoding training. train the temporal weights as a denoising
                autoencoder
            3) CD training, train teh whole model together with CD

        Args:
            train_set_x: the data to train on
            static_epochs: the number of static training epochs
            ae_epochs: the number of autoencoding epochs per delay
            all_epochs: the number of epochs to do CD on the whole model
            lr: the starting learning rate
            k: the number of CD steps to use
            batch_size: size of the minibatches
            dbn_filename: the filename to save this dbn as
            fig_filename: the prefix for all figures to be saved
            plot_interval: the interval at which to plot the model weights.
                0 means no plotting
            save_interval: how many epochs between saves (if dbn_filename set)
            seq_len: the length of each sequence in the data. 
                can be a list if sequences have different lengths
            with_W: whether or not to update the W weight when doing training
                on the whole model. Sometimes we may want to keep the W weights
                from the static training. form vs motion
            binary: if true, treat the visible layer as binary
        """
        #TODO only works for one layer sooo far
        #########################
        # PRETRAINING THE MODEL #
        #########################
        if not isinstance(train_set_x,
                          theano.tensor.sharedvar.TensorSharedVariable):
            train_set_x = theano.shared(train_set_x)

        # Pre-train layer-wise
        for layer in xrange(self.n_layers):
            if static_epochs > 0:
                self._do_static_training(layer, train_set_x, static_epochs, lr, k,
                            batch_size, binary, dbn_filename, fig_filename,
                            save_interval, plot_interval)

            permindex, n_train_batches = self._get_temporal_batches(seqlen,
                            train_set_x.get_value(borrow=True).shape[0],
                            batch_size)
            if ae_epochs > 0:
                self._do_ae_training(permindex, n_train_batches, layer,
                            train_set_x, ae_epochs, lr,
                            batch_size, binary, dbn_filename, fig_filename,
                            save_interval, plot_interval)
            if all_epochs > 0:
                self._do_all_training(permindex, n_train_batches, layer,
                            train_set_x, all_epochs, lr, k,
                            batch_size, binary, with_W,
                            dbn_filename, fig_filename,
                            save_interval, plot_interval)

            if all_epochs > 0 and plot_interval > 0 and fig_filename != None:
                fig_name = fig_filename + '_finished'
                for d in range(self.delay):
                    self.rbm_layers[layer].plot_weights(fig_name, self.n_ins,
                                            with_tree=False, delay=d + 1)
#        #if i < self.n_layers:
#        fn = self.propup(train_set_x, layer=i, static=True)
#        for batch_index in xrange(n_train_batches):
#            data_idx = permindex[batch_index * batch_size:(batch_index + 1) \
#                             * batch_size]
#
#            propup = np.array(pretraining_fn(data_idx))
#                    # propup for next layer training
#

        
        ## Pre-train layer-wise
        

            

        

    def generate(self, data, data_idx=None, n_samples=1, n_gibbs=30):
        # TODO: Currently only works for 1 layer DBNs.
        # need to propup and sample at top layer
        if data_idx == None:
            data_idx = np.array([self.delay])

        orig_data = np.asarray(data[data_idx], dtype=theano.config.floatX)
        hist_idx = np.array([data_idx - n for n in xrange(0, self.delay)]).T
        hist_idx = hist_idx.ravel()
        hist_data = np.asarray(
                             data[hist_idx].reshape(
                                 (len(data_idx),
                                  self.delay * np.prod(self.n_ins))),
                                     dtype=theano.config.floatX)

        generated_series = self.rbm_layers[0].generate(orig_data, hist_data,
                                    n_samples, n_gibbs)

        generated_series = np.concatenate(
                            (hist_data.reshape(len(data_idx),
                                self.delay,
                                np.prod(self.n_ins))[:, ::-1, :],
                                       generated_series), axis=1)
        return generated_series
