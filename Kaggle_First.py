from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np 
import pandas as pd
from matplotlib import pyplot as plt 

# 获取和读取数据集
train_data = pd.read_csv('/home/lifulin/data/train.csv')
test_data = pd.read_csv('/home/lifulin/data/test.csv')
# 训练数据集包括1460个样本，80个特征，1个标签
# 测试数据集包括1459个样本，80个特征
print(train_data.shape, test_data.shape)
# print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# print(test_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
# 样本ID特征为第0列，将有效的79个特征按样本连结 形成 2919×79 的all_features
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)

# 数据预处理

# 连续型数值的预处理：
# 取所有连续数值的特征索引 
# numeric_features = Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
#       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
#       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
#       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
#       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
#       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
#       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
#       'MoSold', 'YrSold'],
#       dtype='object')
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# 对连续数值的特征做标准化(Standardization):将某一特征的每个值减去该特征在整个数据集上的均值X.mean()再除以再整个数据集上的标准差X.std()
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 对于缺失的连续数值使用平均值填充
all_features = all_features.fillna(all_features.mean())


# 离散型样本特征值的预处理：
# 将离散数值转成指示特征。举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning_RL和MSZoning_RM，其值为0或1。
# 如果一个样本原来在MSZoning里的值为RL，那么有MSZoning_RL=1且MSZoning_RM=0。
# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na = True)
# all_features.shape = (2919, 331), 特征数从79增加到了331

# 通过values属性得到NumPy格式的数据，并转成NDArray方便后面的训练
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))

# 训练模型
# 使用基本的线性回归模型和平方损失函数来训练模型
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net

def log_rmse(net, features, labels):
    # 将小于1的值设成1，使得取对数时数值更稳定
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', 
        {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# K折交叉验证-它返回第i折交叉验证时所需要的训练和验证数据
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k  # // 为整除
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # print(idx)
        # slice(0, 292, None)
        # slice(292, 584, None)
        # slice(584, 876, None)
        # slice(876, 1168, None)
        # slice(1168, 1460, None)
        X_part, y_part = X[idx, :], y[idx]
        # X_part.shape = (292, 331) y_part.shape = (292, 1)
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid
# 在K折交叉验证中我们训练 K 次并返回训练和验证的平均误差。
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        # *data是用来接受任意多个参数并将其放在一个元组中。
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            plt.semilogy(range(1, num_epochs + 1), train_ls)
            plt.xlabel('epochs')
            plt.ylabel('rmse')
            plt.semilogy(range(1, num_epochs + 1), valid_ls)
            plt.legend(['train', 'valid'])
            plt.show()
        print('fold %d, train rmse: %f, valid rmse: %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse: %f, avg valid rmse: %f'
      % (k, train_l, valid_l))

def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    plt.semilogy(range(1, num_epochs + 1), train_ls)
    plt.xlabel('epochs')
    plt.ylabel('rmse')
    plt.show()
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)