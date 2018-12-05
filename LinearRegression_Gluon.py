from mxnet import nd, autograd

# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale = 0.01, shape = labels.shape)

# 读取数据
from mxnet.gluon import data as gdata
batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取mini-batch数据
data_iter = gdata.DataLoader(dataset, batch_size, shuffle = True)


# 定义模型  
from mxnet.gluon import nn
# nn是neural network的缩写, 该模块定义了大量神经网络的层
net = nn.Sequential() # net是一个Sequential实例, Sequential实例可以看作是一个串联各个层的容器
# 全连接层是一个Dense实例, 输出个数为1; 无需指定每一层输入的形状，例如线性回归的输入个数，模型会自动推断
net.add(nn.Dense(1))


# 初始化模型参数
from mxnet import init
# init模块中提供了模型参数初始化的各种方法
# 指定权重参数每个元素将在初始化时随机采样于均值为0标准差为0.01的正态分布. 偏差参数默认初始化为0
net.initialize(init.Normal(sigma=0.01)) 


# 定义损失函数
from mxnet.gluon import loss as gloss
# loss模块中定义了各种损失函数
loss = gloss.L2Loss() # 平方损失又称 L2 范数损失


# 定义优化算法
from mxnet import gluon
# 创建一个Trainer实例，并指定学习率为 0.03 的小批量随机梯度下降（sgd）为优化算法。
# 该优化算法将用来迭代net实例所有通过add函数嵌套的层所包含的全部参数。这些参数可以通过collect_params函数获取。
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})


# 训练模型
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record(): 
            l = loss(net(X), y)
        l.backward()
        #如果将l = loss(net(X), y)替换成l = loss(net(X), y).mean()，就需要将trainer.step(batch_size)相应地改成trainer.step(1)
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

# 从net获得需要的层，并访问其权重（weight）和偏差（bias
dense = net[0]
print(true_w, dense.weight.data())
print(true_b, dense.bias.data())