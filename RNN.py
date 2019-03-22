from mxnet import nd
import random
import zipfile
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

with zipfile.ZipFile('./data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
# print(corpus_chars[:40])
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0 : 10000]

# set()的作用是创造一个集合，同时具有去重的作用。
idx_to_char = list(set(corpus_chars)) #inx_to_char 是根据index找char
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)]) # char_to_idx 是根据char找idx
# print(char_to_idx)
vocab_size = len(char_to_idx)
# print(vocab_size)

# 将corpus中的每个字转化为index
corpus_indices = [char_to_idx[char] for char in corpus_chars]
# print(len(corpus_indices))
sample = corpus_indices[:20]
# print('chars: ', ''.join([idx_to_char[idx] for idx in sample]))
# print('indices: ', sample)

# batch_size指每个小批量的样本数，num_steps为每个样本所包含的时间步数。
# 随机采样中，每个样本是原始序列上任意截取的一段序列，相邻的两个随机小批量在原始序列上的位置，不一定相毗邻
def data_iter_random(corpus_indices, batch_size, num_steps, ctx = None):
    # print(len(corpus_indices))
    # 减1是因为输出的索引是相应输入的索引加1
    num_examples = (len(corpus_indices) - 1) // num_steps
    # print(num_examples)
    epoch_size = num_examples // batch_size
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    def _data(pos):
        return corpus_indices[pos: pos + num_steps]
    
    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i: i + batch_size]
        X = [_data( j * num_steps ) for j in batch_indices]
        Y = [_data( j * num_steps + 1) for j in batch_indices]
        yield nd.array(X, ctx), nd.array(Y, ctx)

# my_seq = list(range(30))
# for X, Y in data_iter_random(my_seq, batch_size = 2, num_steps = 6):
#     print('X: ', X, '\nY: ', Y, '\n')

# 令相邻的两个随机小批量在原始序列上的位置相毗邻 
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx = None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = batch_len - 1 // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

# my_seq = list(range(30))
# for X, Y in data_iter_consecutive(my_seq, batch_size = 3, num_steps = 3):
#     print('X: ', X, '\nY: ', Y, '\n')

# X.T 是X的转置矩阵，每次采样的小批量形状为(batch_size, num_step)
# to_one_hot将这样的小批量变换为可以输入进网络的形状为(batch_size, vocab_size)的矩阵
# 矩阵个数等于num_step
def to_onehot(X, size):
    return  [nd.one_hot(x, size) for x in X.T]

# print(nd.array([0, 2]))
# print(nd.one_hot(nd.array([0, 2]), vocab_size))

# X = nd.arange(10).reshape(2, 5)

# inputs = to_onehot(X, vocab_size)
# print(inputs)
# print(len(inputs), inputs[0].shape)


# 初始化模型参数
# num_hiddens是一个超参数
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape)
    # 隐藏层参数
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens)
    # 输出层参数
    W_hq = _one((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs)
    # 附上梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params

# 定义模型
def init_rnn_state(batch_size, num_hiddens):
	return (nd.zeros(shape = (batch_size, num_hiddens)),)

# rnn函数定义了在一个时间步里如何计算隐藏状态和输出
def rnn(inputs, state, params):
	# inputs和outputs皆为num_steps个形状为(batch_size, vocab_size)的矩阵
	W_xh, W_hh, b_h, W_hq, b_q = params
	H, = state
	outputs = []
	for X in inputs:
		H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
		Y = nd.dot(H, W_hq) + b_q
		outputs.append(Y)
	return outputs, (H,)

X = nd.arange(10).reshape(2, 5)
state = init_rnn_state(X.shape[0], num_hiddens)
inputs = to_onehot(X, vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
print(len(outputs), outputs[0].shape, state_new[0].shape)