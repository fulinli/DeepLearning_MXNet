from mxnet import nd
from mxnet.gluon import nn

# Block类是nn模块里提供的一个模型构造类，我们可以继承它来定义我们想要的模型。
# MLP类继承了Block类构造多层感知机，MLP类重载了Block类的__init__函数和forward函数
class MLP(nn.Block):
    # 声明带有模型参数的层，这里声明了两个全连接层
    # **kwargs表示关键字参数，它是一个dict
    def __init__(self, **kwargs):
        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数参数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu') # 隐藏层
        self.output = nn.Dense(10) # 输出层

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.output(self.hidden(x))

X = nd.random.uniform(shape=(2, 20))
print(X)
net = MLP()
net.initialize()
print(net(X))