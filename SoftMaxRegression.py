import gluonbook as gb
from mxnet import autograd, nd
from matplotlib import pyplot as plt

# 获取和读取数据
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()

def show_fashion_mnist(images, labels):
    #gb.use_svg_display()
    # 这里的 _ 表示我们忽略（不使用）的变量。
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    # zip() 函数用于将可迭代对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象。
    # 如果各个可迭代对象的元素个数不一致，则返回的对象长度与最短的可迭代对象相同。
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# SoftMax运算 将将输入特征与权重线性叠加的结果转换成和为1的合理概率分布
def softmax(X):
    X_exp = X.exp()
    # 只对其中同一列(axis=0)或同一行(axis=1)的元素求和，并在结果中保留行和列这两个维度(keepdims=True)
    partition = X_exp.sum(axis = 1, keepdims = True)
    return X_exp / partition

# 定义损失函数-交叉熵损失函数 SoftMax运算可以得到合理的概率分布, 将其与真实标签的概率分布进行交叉熵运算，得到合理的损失函数
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()

# 定义模型 返回结果为y_hat
def net(X):
    # -1的意思是, 我不知道可以分成多少行, 但是我需要是分成num_inputs列
    return softmax(nd.dot(X.reshape(-1, num_inputs), W) + b) 




# 计算分类准确性
def accuracy(y_hat, y):
    # y_hat.argmax(axis=1)返回矩阵y_hat每行中最大元素的索引
    return (y_hat.argmax(axis = 1) == y.astype('float32')).mean().asscalar()

def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


# 训练模型
num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params = None, lr = None, trainer = None):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        # 这个循环执行 60000 / 256 次
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / len(train_iter),
            train_acc_sum / len(train_iter), test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
          batch_size, [W, b], lr)

for X, y in test_iter:
    break

true_labels = gb.get_fashion_mnist_labels(y.asnumpy())
pred_labels = gb.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:10], titles[0:10])