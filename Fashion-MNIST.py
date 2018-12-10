import gluonbook as gb
from mxnet.gluon import data as gdata
import sys
import time
import matplotlib.pyplot as plt


mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)

# 训练集和测试集中每个类别的图像分别为6000, 1000, 因此len(mnist_train)=60000, len(mnist_test) = 10000
print(len(mnist_train), len(mnist_test))

# feature 对应高和宽均为28像素的图像, 每个像素的数值为0-255之间的8位无符号整数(unit8). 使用三维NDArray存储
feature, label = mnist_train[0]

print(feature.shape, feature.dtype)

print(label, type(label), label.dtype)


# 将数值标签转成相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# 定义可以在一行里画出多个图像和对应标签的函数
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

# 显示训练集中0-11号图像
X, y = mnist_train[0:12]
show_fashion_mnist(X, get_fashion_mnist_labels(y))

batch_size = 256
# Vision Transforms: Transforms can be used to augment input data during training. You can compose multiple transforms sequentially
# ToTensor: Converts an image NDArray to a tensor NDArray.
# 通过ToTensor类将图像数据从 uint8 格式变换成 32 位浮点数格式，并除以 255 使得所有像素的数值均在 0 到 1 之间。
# ToTensor类还将图像通道从最后一维移到最前一维来方便之后介绍的卷积神经网络计算。
transformer = gdata.vision.transforms.ToTensor()

# Gluon的DataLoader允许使用多进程来加速数据读取(暂不支持 Windows 操作系统)
# 通过参数num_workers来设置4个进程读取数据。
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

# transform_first(fn, lazy=True): Returns a new dataset with the first element of each sample transformed by the transformer function fn.
# 通过数据集的transform_first函数，我们将ToTensor的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上。
# class mxnet.gluon.data.DataLoader(dataset, batch_size=None, shuffle=False, sampler=None, last_batch=None, batch_sampler=None, 
#                                   batchify_fn=None, num_workers=0, pin_memory=False, prefetch=None)
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), 
                              batch_size, shuffle=True, 
                              num_workers=num_workers)
# print(train_iter)

test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), 
                            batch_size, shuffle=False, 
                            num_workers=num_workers)
# print(test_iter)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))