import pylab as plt
import sys
sys.path.append('..')
from models.tadbn import TADBN
import numpy
import numpy as np
import theano


test_data = np.array([np.sin(np.arange(400) * 0.2),
                      np.sin(np.arange(400) * 0.4)]).T


batchdata = numpy.asarray(test_data, dtype=theano.config.floatX)
delay = 3

numpy_rng = numpy.random.RandomState(123)
n_dim = [test_data.shape[1]]

dbn_tadbn = TADBN(numpy_rng=numpy_rng, n_ins=[n_dim],
          hidden_layers_sizes=[10],
          sparse=0.0, delay=delay, learning_rate=0.01)

up = dbn_tadbn.propup(batchdata, static=False)



up = np.array(up)
print up.shape, batchdata.shape


