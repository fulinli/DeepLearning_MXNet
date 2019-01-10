from matplotlib import pyplot as plt
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5

features = nd.random.normal(shape=(n_train + n_test, 1))

print(features.shape)

poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))

print(poly_features.shape)

labels = true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b 

labels += nd.random.normal(scale=0.01, shape=labels.shape)

print(labels.shape)

print(features[:2], poly_features[:2], labels[:2])

plt.semilogy(features[:,0].asnumpy(), labels[:].asnumpy(), linestyle=':')
plt.show()