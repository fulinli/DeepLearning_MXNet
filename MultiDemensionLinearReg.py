from matplotlib import pyplot as plt 
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import data as gdata, loss as gloss, nn

# y = 0.05 + 0.01*x1 + ... + 0.01*xp + bias

# 生成数据集
n_train, n_test, num_inputs = 20, 100, 200

true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

# 初始化模型参数
def init_params():
    w = nd.random.normal(scale = 1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]

# 定义L2范数惩罚项
def l2_penalty(w):
    return (w**2).sum() / 2

# 定义训练和测试
batch_size, num_epochs, lr = 1, 100, 0.005

def net(X, w, b):
    return nd.dot(X, w) + b

def loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().asscalar())
    plt.semilogy(range(1, num_epochs + 1), train_ls)
    plt.semilogy(range(1, num_epochs + 1), test_ls, linestyle=':')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
    plt.show()
    print('L2 norm of w: ', w.norm().asscalar())

def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd', {'learning_rate': lr, 'wd': wd})
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd', {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    plt.semilogy(range(1, num_epochs + 1), train_ls)
    plt.semilogy(range(1, num_epochs + 1), test_ls, linestyle=':')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'test'])
    plt.show()
    print('L2 norm of w: ', net[0].weight.data().norm().asscalar())

fit_and_plot_gluon(3)
