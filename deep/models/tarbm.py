""" Theano TARBM Implementation
@author Chris Hausler - Freie Universitaet Berlin
Adapted from Graham Taylor's CRBM implementation
https://gist.github.com/2505670
"""
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import cPickle
import theano
import theano.tensor as T
from theano.printing import Print
from theano.tensor.shared_randomstreams import RandomStreams
from utils.plotting import tile_raster_images
floatX = theano.config.floatX


class TARBM(object):
    """Temporal Autoencoding Restricted Boltzmann Machine (TARBM)
    """
    def __init__(self, input_=None, input_history=None, n_visible=49,
                 n_hidden=500, delay=0, A=None, W=None, hbias=None,
                 vbias=None, numpy_rng=None,
                 theano_rng=None, sparse=0, lr=None):
        """TARBM constructor.

        Defines the parameters of the model along with
        basic operations for inferring hidden from visible (and vice-versa),
        as well as for performing CD updates.

        Args:
            input: None for standalone RBMs or symbolic variable if RBM is
        part of a larger graph.

        :param n_visible: number of visible units

        :param n_hidden: number of hidden units

        :param A: None for standalone CRBMs or symbolic variable pointing to a
        shared weight matrix in case CRBM is part of a CDBN network; in a CDBN,
        the weights are shared between CRBMs and layers of a MLP

        :param B: None for standalone CRBMs or symbolic variable pointing to a
        shared weight matrix in case CRBM is part of a CDBN network; in a CDBN,
        the weights are shared between CRBMs and layers of a MLP

        :param W: None for standalone CRBMs or symbolic variable pointing to a
        shared weight matrix in case CRBM is part of a CDBN network; in a CDBN,
        the weights are shared between CRBMs and layers of a MLP

        :param hbias: None for standalone CRBMs or symbolic variable pointing
        to a shared hidden units bias vector in case CRBM is part of a
        different network

        :param vbias: None for standalone RBMs or a symbolic variable
        pointing to a shared visible units bias
        """

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.delay = delay
        self.sparse = sparse
        self.binary = False
        if numpy_rng is None:
            # create a number generator
            numpy_rng = np.random.RandomState(1234)

        if theano_rng is None:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        if W is None:
            # the output of uniform if converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = np.asarray(0.01
                        * numpy_rng.randn(n_visible, n_hidden), dtype=floatX)
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W')

        self.reset_A(A)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(value=np.zeros(n_hidden,
                                dtype=floatX), name='hbias')

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(value=np.zeros(n_visible,
                                dtype=floatX), name='vbias')

        # initialize input layer for standalone CRBM or layer0 of CDBN
        self.input = input_
        if not input:
            self.input = T.matrix('input')

        self.lr = lr
        if not lr:
            self.lr = T.dscalar('lr')

        self.input_history = input_history
        if not input_history:
            self.input_history = T.matrix('input_history')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.A, self.hbias, self.vbias]

    def reset_A(self, A=None):
        """ Resets the temporal Hidden to Hidden weights

        Args:
            A: if None, set the weights randomly, otherwise use A
        """
        if A is None:
            #  we only need the temporal weights if the model has a delay
            if self.delay > 0:
                initial_A = np.asarray(0.01 * np.random.randn(self.delay,
                                            self.n_hidden, self.n_hidden),
                                            dtype=floatX)
            else:
                initial_A = None
            # theano shared variables for weights and biases
            A = theano.shared(value=initial_A, name='A')
        self.A = A

    def free_energy(self, vis, v_history):
        """ Function to compute the free energy of data  presented to the model

        Args:
            vis: an theano tensor containing the values of the visible
                layer
            v_history: a theano tensor containg the values of the visible layer
                at past time steps

        Returns: the free energy
        """
        static = self.delay > 0
        wx_b, _ = self.propup(vis, v_history, static=static)
        ax_b = self.vbias

        visible_term = T.sum(0.5 * T.sqr(vis - ax_b), axis=1)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return visible_term - hidden_term

    def propup(self, vis, v_history=None, static=False, delay=None,
               past_sample=False):
        """ This function propagates the visible units activation upwards to
        the hidden units

        Args:
            vis: theano tensor of the visible layer activation to propagate
            v_history: the visible unit activation for past time steps,
                if required
            static: if static, ignore any temporal components (normal RBM)
            delay: the number of timesteps in the past to consider for,
                if None, use the models full delay
            past_sample: if True, take a binomial sample from past hidden
                layers, otherwise use the mean field activation

        Returns:
            Note that we return also the pre-sigmoid activation of the layer.As
            it will turn out later, due to how Theano deals with optimizations,
            this symbolic variable will be needed to write down a more
            stable computational graph (see details in the reconstruction cost
            function)
        """
        if not delay:
            delay = self.delay
        if (delay > 0) and (not static):
            pre_sigmoid_activation = T.dot(vis, self.W)
            K = np.prod(self.n_visible)
            for i in range(delay):
                [_, hid_act] = self.propup(v_history[:, i * K:(i + 1) * K],
                                          static=True)
                if past_sample:
                    hid_act = self.theano_rng.binomial(size=hid_act.shape, n=1,
                                             p=hid_act,
                                             dtype=floatX)
                pre_sigmoid_activation += T.dot(hid_act, self.A[i])
            pre_sigmoid_activation = pre_sigmoid_activation + self.hbias
        else:
            pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias

        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, vis, v_history, past_sample=False):
        """ This function infers state of hidden units given visible units

        Args:
            vis: theano tensor of the visible layer activation to propagate
            v_history: the visible unit activation for past time steps,
                if required
            past_sample: if True, take a binomial sample from past hidden
                layers, otherwise use the mean field activation
        Returns:
            a list containing the pre-sigmoid, mean field and binomial sample
            for the hidden layer given the input
         """

        pre_sigmoid_h1, h1_mean = self.propup(vis, v_history,
                                              past_sample=past_sample)

        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape, n=1,
                                             p=h1_mean,
                                             dtype=floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """This function propagates the hidden units activation downwards to
        the visible units

        Args:
            hid: the hidden unit activation
        Returns:
            pre_sigmoid visible unit activation
        """
        pre_sigmoid = T.dot(hid, self.W.T) + self.vbias
        #TODO should this really be presigmoid?
        return pre_sigmoid

    def sample_v_given_h(self, hid):
        """ This function infers state of visible units given hidden units
        Args:
            hid: the hidden unit activation
        Returns:
            mean field visible unit activation and a binomial sample of
            the sigmoid(mean_field)
        """
        # compute the activation of the visible given the hidden sample
        #pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_mean = self.propdown(hid)
        v1_sig = T.nnet.sigmoid(v1_mean)
        v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
                                             n=1, p=v1_sig,
                                            dtype=floatX)
        #TODO do we want mean or pre sigmoid?
        return v1_mean, v1_sample

    def gibbs_hvh(self, hid, v_history):
        """ This function implements one step of Gibbs sampling,
            starting from the hidden state

        Args:
            hid: the hidden unit activation
            v_history: the visible unit activation for past time steps,
                if required
        Returns:
            a list containing
            v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample
            after 1 gibbs sample
        """
        v1_mean, v1_sample = self.sample_v_given_h(hid)

        if self.binary:
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(
                                                        v1_sample, v_history)
        else:
            pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_mean,
                                                                   v_history)

        return [v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, vis, v_history, past_sample=False):
        """ This function implements one step of Gibbs sampling,
            starting from the visible state

        Args:
            vis: theano tensor of the visible layer activation to propagate
            v_history: the visible unit activation for past time steps,
                if required
            past_sample: if True, take a binomial sample from past hidden
                layers, otherwise use the mean field activation
        Returns:
            a list containing
            pre_sigmoid_h1, h1_mean, h1_sample, v1_mean, v1_mean
            after one gibbs sample
        """

        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(vis,
                                            v_history, past_sample=past_sample)
        v1_mean, _ = self.sample_v_given_h(h1_sample)

        return [pre_sigmoid_h1, h1_mean, h1_sample, v1_mean, v1_mean]

    def get_cost_updates(self, delay=0, k=1, persistent=False,
                         static=False, with_W=False, past_sample=False):
        """
        This functions implements one step of CD-k
        Args:
            delay: the number of timesteps in the past to consider for,
                    if None, use the models full delay
            k: number of Gibbs steps to do in CD-k
            persistent: None for CD
            static: if static, ignore any temporal components (normal RBM)
            with_W: whether or not to update param W. For movies we may like to
                keep this fixed after the static training. Form vs Motion
            past_sample: if True, take a binomial sample from past hidden
                layers, otherwise use the mean field activation

        Returns:
            a proxy for the cost and the updates dictionary. The
            dictionary contains the update rules for weights and biases but
            also an update of the shared variable used to store the persistent
            chain, if one is used.
        """

        tmp_delay = self.delay
        if static:
            self.delay = 0

        # compute positive phase
        _, ph_mean, ph_sample = \
                        self.sample_h_given_v(self.input, self.input_history,
                                              past_sample)

        # for CD, we use the newly generate hidden sample
        chain_start = ph_sample

        # perform actual negative phase
        # in order to implement CD-k we need to scan over the
        # function that implements one gibbs step k times.
        # Read Theano tutorial on scan for more information :
        # http://deeplearning.net/software/theano/library/scan.html
        # the scan will return the entire Gibbs chain
        # updates dictionary is important because it contains the updates
        # for the random number generator
        #TODO check how many Nones should be here ...
        [nv_means, nv_samples, _, _, _], updates = theano.scan(self.gibbs_hvh,
                    # the None are place holders, saying that
                    # chain_start is the initial state corresponding to the
                    # 5th output
                    outputs_info=[None, None, None, None, None, chain_start],
                    non_sequences=self.input_history,
                    n_steps=k)

        # determine gradients on RBM parameters
        # not that we only need the sample at the end of the chain
        if self.binary:
            chain_end = nv_samples[-1]
        else:
            chain_end = nv_means[-1]

        cost = T.mean(self.free_energy(self.input, self.input_history)) - \
               T.mean(self.free_energy(chain_end, self.input_history))

        # if static or no delay, ignore temporal weights
        if static and tmp_delay > 0:
            prms = [self.params[0], self.params[2], self.params[3]]
        # sometimes we don't want change what we learnt in static pretraining,
        # so ignore the vis bias and W weights
        elif not with_W:
            prms = [self.params[1], self.params[2]]
        else:
            prms = self.params

        # We must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, prms, consider_constant=[chain_end],
                         disconnected_inputs='warn')

        # constructs the update dictionary
        for gparam, param in zip(gparams, prms):
            if param == self.A:
                # slow down autoregressive updates
                updates[param] = param - gparam * 0.01 * \
                                  T.cast(self.lr, dtype=floatX)
            # enforce sparsity on hidden bias if a sparse target is set
            elif (self.sparse != 0 and param == self.hbias):
                updates[param] = param + (
                            (T.cast(self.sparse, dtype=floatX)
                                        - ph_mean.mean(0))
                        * T.cast(self.lr * 0.1, dtype=floatX))
            else:
                updates[param] = param - gparam * \
                            T.cast(self.lr, dtype=floatX)

        # reconstruction error is a better proxy for CD
        #monitoring_cost = self.get_reconstruction_cost(updates, nv_means[-1])

        # if this was a static run, reset the delay of the model
        if static:
            self.delay = tmp_delay

        return cost, updates

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        """Approximation to the reconstruction error

        Args:
            pre_sigmoid_nv: pre sigmoid visible value from sample
        Returns:
            the means squared construction error
        """
        # sum over dimensions, mean over cases
        recon = T.mean(T.sum(T.sqr(self.input - pre_sigmoid_nv), axis=1))

        return recon

#TODO remove this code if not needed
#    def get_hidden_values(self, input, delay):
#        """ Computes the values of the hidden layer at t=0 based on previous
#        timestep
#        """
#        return T.nnet.sigmoid(T.dot(input, self.A[delay]) + self.hbias)

    def get_cost_updates_AE(self, delay, corruption,
                             sparsity=0.05, sparse_weight=0.0):
        """ This function computes the cost and the updates for one training
            step of the autoencoder reconstruction

        This function is used to Autoencode the temporal weights
        Parts of the cost function calculation is stolen from
        the PyLearn toolkit. Check it out, it's very nice!
        https://github.com/lisa-lab/pylearn

        Args:
            delay: which delay are we working on
            corruption: how much corruption has been added to the input signal
            sparsity: the sparsity target for the hidden layer
            sparse_weight: how to weight the sparsity target by in the cost

        Returns:
            cost and update
        """
        curr_hid = None
        K = np.prod(self.n_visible)
        for d in range(delay + 1):
            _, hist_hid = self.propup(self.input_history[:, d * K:(d + 1) * K],
                                      static=True)
            if not curr_hid:
                curr_hid = T.dot(hist_hid, self.A[d])
            else:
                curr_hid += T.dot(hist_hid, self.A[d])
        curr_hid = T.nnet.sigmoid(curr_hid + self.hbias)

        reconstruction = self.propdown(curr_hid)
        if self.binary:
            reconstruction = T.nnet.sigmoid(reconstruction)
            L = -T.sum(self.input * T.log(reconstruction) + (1 - self.input)
                       * T.log(1 - reconstruction), axis=1)
            cost = T.mean(L)

        else:
            model_score = -(self.input - reconstruction)  # / T.sqr(self.sigma)
            model_score.name = 'score'

            axis = range(1, len(self.input.type.broadcastable))
            corruption_free_energy = (T.sum(T.sqr(self.input_history
                                [:, delay * K:(delay + 1) * K] - self.input),
                                 axis=axis) / (2. * (corruption ** 2.)))

            corruption_free_energy = (Print('corruption_free_energy')
                                      (corruption_free_energy))
            parzen_score = T.grad(-T.sum(corruption_free_energy),
                                  self.input_history)
            #parzen_score = Print('parzen_score')(parzen_score)

            score_diff = model_score - parzen_score[:,
                                                    delay * K:(delay + 1) * K]
            sq_score_diff = T.sqr(score_diff)
            #sq_score_diff = (Print('sq_score_diff',attrs=['mean'])
            #                       (sq_score_diff))
            smd = T.mean(sq_score_diff)
            cost = smd

        if sparse_weight > 0:
            sparse_penalty = T.sum(sparsity
                                   * T.log(sparsity / curr_hid.mean(axis=0))
                                   + (1 - sparsity)
                                   * T.log((1 - sparsity)
                                           / (1 - curr_hid.mean(axis=0))))
            cost += sparse_weight * sparse_penalty

        theano.printing.Print('this is a very important value')(cost)

#        # note : L is now a vector, where each element is the
#        #        cross-entropy cost of the reconstruction of the
#        #        corresponding example of the minibatch. We need to
#        #        compute the average of all these to get the cost of
#        #        the minibatch
#        cost = T.mean(L)
#
        # compute the gradients of the cost of the `cA` with respect
        # to its parameters
        prms = [self.A, self.hbias]
        gparams = T.grad(cost, prms)
        # generate the list of updates
        updates = {}
        for param, gparam in zip(prms, gparams):
            #gparam = Print('gparam',attrs=['mean(axis=0)'])(gparam)
            updates[param] = param - T.cast(self.lr,
                                        dtype=floatX) * gparam
        return cost, updates

    def generate(self, orig_data, orig_history, n_samples, n_gibbs=30):
        """ Given initialization(s) of visibles and matching history, generate
        n_samples in future.

        Args:
            orig_data: n_seq by n_visibles array
                initialization for first frame
            orig_history: n_seq by delay * n_visibles array
                delay-step history
            n_samples: int
                number of samples to generate forward
            n_gibbs: int
                number of alternating Gibbs steps per iteration
        Returns:
            generated samples for each of the input sequences
        """

        #TODO Try sampling from the past values as well. binary not real value

        n_seq = orig_data.shape[0]
        persistent_vis_chain = theano.shared(orig_data)
        persistent_history = theano.shared(orig_history)

        # do the generation
        [_, _, _, vis_mfs, vis_samples], updates = theano.scan(self.gibbs_vhv,
                                    outputs_info=[None, None, None, None,
                                                    persistent_vis_chain],
                                    non_sequences=persistent_history,
                                    n_steps=n_gibbs)

        # add to updates the shared variable that takes care of our persistent
        # chain
        # initialize next visible with current visible
        # shift the history one step forward
        updates.update({persistent_vis_chain: vis_samples[-1],
                         persistent_history: T.concatenate(
                             (vis_samples[-1],
                                 persistent_history[:, :(self.delay - 1) * \
                                                    self.n_visible],
                              ), axis=1)})

        # construct the function that implements our persistent chain.
        # we generate the "mean field" activations for plotting and the actual
        # samples for reinitializing the state of our persistent chain
        sample_fn = theano.function([], [vis_mfs[-1], vis_samples[-1]],
                            updates=updates,
                            name='sample_fn')  # , mode='DEBUG_MODE')

        generated_series = np.empty((n_seq, n_samples, self.n_visible))
        for t in xrange(n_samples):
            print "Generating frame %d" % t
            vis_mf, _ = sample_fn()
            generated_series[:, t, :] = vis_mf
        return generated_series

    ##################### plotting stuffs
    #TODO. this all needs to be cleaned up. commented. made more useful
    def plot_weights(self, fname=None, img_shape=(8, 8), rgb=False,
                     static=False, with_tree=False, delay=0):
        tile = int(np.ceil(np.sqrt(self.n_hidden)))
        X = self.W.get_value(borrow=True).T
        K = np.prod(img_shape)
        A = self.A.get_value(borrow=True)

        if len(img_shape) > 1:
            f = plt.figure()

            if rgb:

                X = (X[:, :K], X[:, K:2 * K], X[:, 2 * K:], None)
            out = tile_raster_images(X, img_shape, (tile, tile), (1, 1),
                                      scale_rows_to_unit_interval=True)
            plt.imshow(out, cmap=plt.cm.gray, interpolation='none')

            if not fname:
                plt.show()
            else:
                f.savefig(fname + '.png')
            plt.close(f)

        f = plt.figure(figsize=(16, 16))
        plt.subplot(231)
        plt.hist(self.W.get_value(borrow=True).ravel(), 100)
        plt.subplot(232)
        plt.hist(self.vbias.get_value(borrow=True).ravel(), 100)
        plt.subplot(233)
        plt.hist(self.hbias.get_value(borrow=True).ravel(), 100)
        if A is not None:
            plt.subplot(212)
            plt.hist(A.ravel(), 100)
        if not fname:
            plt.show()
        else:
            f.savefig(fname + '_hist.png')

        plt.close(f)

        spacer = np.zeros(np.prod(img_shape))[np.newaxis, :]
        if (A is not None) and not static and len(img_shape) > 1:
            all_units = None
            for i in range(self.n_hidden):
                if with_tree:
                    self.history_tree_plot(i, fname, img_shape, 3,
                                           delays=delay)

                units = []
                images = []
                for _ in range(delay + 1):
                    units.append([])
                    images.append([])
                self.unit_activation_trace(units, images, 1, delay + 1, [i])
                unit_plot = np.array(images)[:, 0, :]
                if all_units is None:
                    all_units = unit_plot
                else:
                    all_units = np.append(all_units, unit_plot, axis=0)
                all_units = np.append(all_units, spacer, axis=0)

            f = plt.figure(figsize=(16, 16))
            plts = delay + 2
            total = self.n_hidden * plts
            tile = tile2 = int(np.ceil(np.sqrt(total)))
            if tile % plts != 0:
                tile += plts - (tile % plts)
            out = tile_raster_images(all_units, img_shape, (tile2, tile),
                        (1, 1), scale_rows_to_unit_interval=True)
            plt.imshow(out, cmap=plt.cm.gray, interpolation='nearest')
            #plt.show()
            f.savefig(fname + '_temp_hist_delay_' + str(delay) + '.png')
            plt.close(f)

    def unit_activation_trace(self, units, images, num_samples, max_delay,
                              parents):
        if len(parents) == max_delay:
            return

        A = self.A.get_value(borrow=True)
        W = self.W.get_value(borrow=True).T

        if len(parents) == 1:
            units[0].append(parents[0])
            images[0].append(W[parents[0]])

        activation = np.zeros(A.shape[2])
        for i, u in enumerate(parents):
            activation += A[-(i + 1), u, :]

        idxs = np.argsort(activation)[::-1]
        idxs = idxs[:num_samples].tolist()
        units[len(parents)] += idxs
        for s in idxs:
            tmp_par = [s] + parents
            images[len(parents)].append(W[s])
            self.unit_activation_trace(units, images, num_samples, max_delay,
                              tmp_par)

    def history_tree_plot(self, hidden_unit, fname, img_shape=(8, 8),
                    img_samples=3, delays=0, x_spacer=3,  y_spacer=6,
                    group_spacer=8, group_group_spacer=8):

        img_x = img_shape[0]
        img_y = img_shape[1]
        num_plots = img_samples ** delays
        num_groups = num_plots / img_samples

        units = []
        images = []
        for i in range(delays + 1):
            units.append([])
            images.append([])

        f = plt.figure(figsize=(16, 16))
        strt = 1
        last_group_x = []
        for g in range(num_groups):
            group_pos = []
            for s in range(img_samples):
                group_pos.append(strt)
                strt += img_x + x_spacer
            last_group_x.append(group_pos)
            strt += group_spacer - x_spacer
            if (g + 1) % img_samples == 0:
                strt += group_group_spacer

        X = strt + 1 - group_spacer
        Y = (delays + 1) * img_y + delays * y_spacer + 2

        img = np.zeros((delays + 1, X, Y))
        img[:] = None

        y = 0
        for d in range(delays + 1):
            plt_num = 0
            new_group_x = []
            xs = []
            for g in last_group_x:
                for i, s in enumerate(g):
                    im = images[-(d + 1)][plt_num]
                    im = np.reshape(im, img_shape)
                    img[d, s:s + img_x, y:y + img_y] = im
                    plt_num += 1
                    if i == np.floor(img_samples / 2.):
                        xs.append(s)
                        if len(xs) == img_samples:
                            new_group_x.append(xs)
                            xs = []
            if len(xs) > 0:
                new_group_x.append(xs)
            last_group_x = new_group_x
            y += img_y + y_spacer
        pwargs = {'cmap':plt.cm.gray, 'interpolation':'nearest',
                  'origin':'lower'}
        for ii in img:
            plt.imshow(ii.T, **pwargs)

        if not fname:
            plt.show()
        else:
            fname += '_temp_hist_delay_%d_unit_%d.png' % (delays, hidden_unit)
            f.savefig(fname)
        plt.close(f)

    def save_evolution(self, fname):
        all_units = None
        for i in range(self.n_hidden):
            units = []
            images = []
            for _ in range(self.delay + 1):
                units.append([])
                images.append([])
            self.unit_activation_trace(units, images, 3, self.delay + 1, [i])
            f = open(fname + '_ev_unit_%d.pkl' % i, 'wb')
            cPickle.dump((units, images), f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()

            units = []
            images = []
            for _ in range(self.delay + 1):
                units.append([])
                images.append([])
            self.unit_activation_trace(units, images, 1, self.delay + 1, [i])
            unit_plot = np.array(images)[np.newaxis, :, 0, :]
            if all_units is None:
                all_units = unit_plot
            else:
                all_units = np.append(all_units, unit_plot, axis=0)
        f = open(fname + '_ev_all_units.pkl', 'wb')
        cPickle.dump(all_units, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
