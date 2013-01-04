import pylab as plt


from tadbn import TADBN
import numpy
import numpy as np
import theano



data = np.array([np.sin(np.arange(400)*0.1), np.sin(np.arange(400)*0.2)]).T
print data.shape
#plt.plot(data)
#plt.show()
batchdata = theano.shared(numpy.asarray(data,
                                               dtype=theano.config.floatX))
    
delay = 4
# numpy random generator
numpy_rng = numpy.random.RandomState(123)
n_dim = [1,2]

dbnTRBM = TADBN(numpy_rng=numpy_rng, n_ins=[n_dim],
          hidden_layers_sizes=[10],
          n_outs=0, sparse=0.0, delay=delay, learning_rate=0.01)
    
#dbnTRBM.pretrain(batchdata, 
#              plot_interval = 5, 
#              static_epochs = 30, save_interval=10,
#              epochs = 30, all_epochs=10, batch_size = 5)
#
#
#dbnCRBM = cDBN(numpy_rng=numpy_rng, n_ins=[n_dim],
#          hidden_layers_sizes=[10],
#          n_outs=0, sparse=0.0, delay=delay, learning_rate=0.01)
##
#dbnCRBM.pretrain(batchdata,                   
#              static_epochs = 30,
#              epochs = 200, batch_size = 5)
    

data_idx = np.array([dbnTRBM.delay])
orig_data = numpy.asarray(data[data_idx],
                              dtype=theano.config.floatX)
hist_idx = np.array([data_idx - n for n in xrange(1, dbnTRBM.delay + 1)]).T    
print hist_idx
hist_idx = hist_idx.ravel()

n_samples = 40
orig_history = numpy.asarray(
        data[hist_idx].reshape(
        (len(data_idx), dbnTRBM.delay * np.prod(dbnTRBM.n_ins))),
        dtype=theano.config.floatX)


print orig_history.shape
generated_seriesTRBM = dbnTRBM.rbm_layers[0].generate(orig_data, orig_history, n_samples=n_samples,
                                     n_gibbs=30)



generated_seriesTRBM = np.concatenate((orig_history.reshape(len(data_idx),
                                                            dbnTRBM.delay,
                                                            np.prod(dbnTRBM.n_ins) \
                                                            )[:, ::-1, :],
                                       generated_seriesTRBM), axis=1)


generated_seriesCRBM = dbnCRBM.rbm_layers[0].generate(orig_data, orig_history, n_samples=n_samples,
                                     n_gibbs=30)



generated_seriesCRBM = np.concatenate((orig_history.reshape(len(data_idx),
                                                            dbnCRBM.delay,
                                                            np.prod(dbnCRBM.n_ins) \
                                                            )[:, ::-1, :],
                                       generated_seriesCRBM), axis=1)


plt.figure()
plt.subplot(211)
plt.plot(generated_seriesTRBM[0])
plt.subplot(212)
plt.plot(generated_seriesCRBM[0])
plt.show()


