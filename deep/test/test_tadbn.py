import pylab as plt


from models.tadbn import TADBN
import numpy
import numpy as np
import theano


test_data = np.array([np.sin(np.arange(400) * 0.2),
                      np.sin(np.arange(400) * 0.4)]).T

print test_data.shape
batchdata = numpy.asarray(test_data, dtype=theano.config.floatX)

delay = 10

numpy_rng = numpy.random.RandomState(123)
n_dim = [test_data.shape[1]]

dbn_tadbn = TADBN(numpy_rng=numpy_rng, n_ins=[n_dim],
          hidden_layers_sizes=[100],
          n_outs=0, sparse=0.0, delay=delay, learning_rate=0.01)

dbn_tadbn.pretrain(batchdata, plot_interval=5, static_epochs=30,
                   save_interval=10, epochs=30, all_epochs=60,
                   batch_size=5)

generated_series = dbn_tadbn.generate(batchdata, n_samples=40)

plt.figure()
plt.subplot(211)
plt.plot(test_data[:generated_series.shape[1]])
plt.subplot(212)
plt.plot(generated_series[0])
plt.show()


