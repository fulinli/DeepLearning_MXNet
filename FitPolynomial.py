from matplotlib import pyplot as plt
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn


# 生成数据集
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
print(features.shape)
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
labels = true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1] + true_w[2] * poly_features[:, 2] + true_b 
labels += nd.random.normal(scale=0.01, shape=labels.shape)
# 查看数据集的前两个样本
print(features[:2], poly_features[:2], labels[:2])

#plt.scatter(features[:,0].asnumpy(), labels[:].asnumpy())
#plt.show()

num_epochs, loss = 100, gloss.L2Loss()

num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    plt.semilogy(range(1, num_epochs + 1), train_ls)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.semilogy(range(1, num_epochs + 1), test_ls)
    plt.legend(['train', 'test'])
    plt.show()
    #semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             #range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())

fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])