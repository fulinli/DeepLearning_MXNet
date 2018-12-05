from mxnet import autograd, nd
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2

# 设训练数据集样本size为1000*2, 即X1, X2, 均值为0，标准差为1的正态分布
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))

# leables --- Y = w1 * x1 + w2 * x2 + b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 给Y加上噪音项ϵ(服从均值为0, 标准差为0.01的正态分布)
labels += nd.random.normal(scale = 0.01, shape = labels.shape)

# data_iter 从features-labels中选出batch_size大小的小批量数据
# 它每次返回batch_size（批量大小）个随机样本的特征和标签。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 打乱indices中element的顺序
    random.shuffle(indices)
    # range(start_id, end_id, step_size)
    for i in range(0, num_examples, batch_size):
        # 防止样本个数不能被批量大小整除, 因此采用min(i + batch_size, num_examples)
        j = nd.array(indices[i: min(i + batch_size, num_examples)]) 
        # take 函数根据索引返回对应元素。
        yield features.take(j), labels.take(j)
batch_size = 10
# 初始化模型参数, w服从均值为0, 标准差为0.01的正态分布, b初始化为0 
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

# 创建梯度
w.attach_grad()
b.attach_grad()

# 定义模型 
def linreg(X, w, b):
    return nd.dot(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# 训练模型
lr = 0.03  #learning rate
num_epochs = 3  #迭代次数

for epoch in range(num_epochs): 
    n = 0
    for X,y in data_iter(batch_size, features, labels): #先将mini-batch数据取出
        n = n + 1
        print(n)
        with autograd.record():
            # 计算损失函数
            loss_function = squared_loss(linreg(X, w, b), y)
            # 由于我们之前设批量大小batch_size为 10，每个小批量的loss_function的形状为（10，1）
            # 由于变量loss_function并不是一个标量，运行loss_function.backward()将对loss_function中元素求和得到新的变量，再求该变量有关模型参数的梯度
        #计算损失函数梯度
        loss_function.backward()
        #更新w, b
        sgd([w, b], lr, batch_size)
    train_l = squared_loss(linreg(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

print(true_w, w)
print(true_b, b)