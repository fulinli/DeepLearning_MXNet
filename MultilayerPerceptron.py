import gluonbook as gb 
from mxnet import nd
from mxnet.gluon import loss as gloss

batch_size = 256

train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
# 设置超参数-隐藏层单元数为 256
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 128, 256

W1 = nd.random.normal(scale = 0.01, shape = (num_inputs, num_hiddens1)) # W1.shape = (784, 256)
b1 = nd.zeros(num_hiddens1) # b1.shape = (1, 256)
W2 = nd.random.normal(scale = 0.01, shape = (num_hiddens1, num_hiddens2)) # W2.shape = (256, 10)
b2 = nd.zeros(num_hiddens2) # b2.shape = (1, 10)
W3 = nd.random.normal(scale = 0.01, shape = (num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

def relu(X):
    return nd.maximum(X, 0)

def net(X):
    X = X.reshape((-1, num_inputs))
    H1 = relu(nd.dot(X, W1) + b1)
    H2 = relu(nd.dot(H1, W2) + b2)
    return nd.dot(H2, W3) + b3

loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 5, 0.5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)