from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata

def load_data_fashion_mnist(batch_size):
    train_data = gdata.vision.datasets.FashionMNIST(train = True)
    test_data = gdata.vision.datasets.FashionMNIST(train = False)
    transformer = gdata.vision.transforms.ToTensor()
    train_iter = gdata.DataLoader(train_data.transform_first(transformer), batch_size, shuffle = True)
    test_iter = gdata.DataLoader(test_data.transform_first(transformer), batch_size, shuffle = True)
    return train_iter, test_iter


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr):

    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            sgd(params, batch_size, lr)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(net(X), y)
        test_loss = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / len(train_iter),
            train_acc_sum / len(train_iter), test_loss))


def sgd(params, batch_size, lr):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

def accuracy(y_hat, y):
    return (y_hat.argmax(axis = 1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(test_iter, net):
    acc = 0
    for X, y in test_iter:
        acc += accuracy(net(X), y)
    return acc / len(test_iter)

def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        return X.zeros_like()
    # mask = nd.random.uniform(0, 1, X.shape) < keep_prob
    randomX = nd.random.uniform(0, 1, X.shape)
    mask =  randomX < keep_prob
    # print(mask)
    return mask * X / keep_prob

# X = nd.arange(16).reshape((2, 8))
# =print(dropout(X, 0))
# print(dropout(X, 0.5))
# print(dropout(X, 1))

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

W1 = nd.random.normal(scale = 0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
W2 = nd.random.normal(scale = 0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
W3 = nd.random.normal(scale = 0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

drop_prob1, drop_prob2 = 0.2, 0.5

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = (nd.dot(X, W1) + b1).relu()
    if autograd.is_training():
        H1 = dropout(H1, drop_prob1)
    H2 = (nd.dot(H1, W2) + b2).relu()
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)
    return nd.dot(H2, W3) + b3

num_epochs, lr, batch_size = 5, 0.5, 256
loss = gloss.SoftmaxCrossEntropyLoss()
train_iter, test_iter = load_data_fashion_mnist(batch_size)
train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)