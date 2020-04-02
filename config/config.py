import torch
from torch import optim

from enum import unique, Enum


# 梯度下降
@unique
class Opt(Enum):
    ADAM = 'adam'
    SGD = 'sgd'
    MOMENTUM = 'momentum'


# 操作类型
@unique
class PHASE(Enum):
    PREDICT = 'predict'
    TRAIN = 'train'
    TEST = 'test'


# 操作类型
phase = PHASE.TRAIN
# 训练次数
epochs = 10000
# 优化器选择
default_optim = Opt.ADAM
# 训练批次数量
train_batch_size = 10
# 测试批次数量
test_batch_size = 1
# 学习率
learn_rate = 0.001
# 动量
momentum = 0.8
alpha = 0.9
# 模型保存/加载地址
model_path = '../pts/model.pt'
# 日志打印周期
log_interval = 10
# 模型保存周期
save_interval = 10
# 预测图片
predict_image = '../non/I/000007.jpg'
# 常用损失
celoss = torch.nn.CrossEntropyLoss()
mseloss = torch.nn.MSELoss()


# 梯度下降选择
def optimizer(model, _type: Opt = default_optim):
    params = model.parameters()
    if _type is Opt.ADAM:
        return optim.Adam(params, lr=learn_rate)
    if _type is Opt.MOMENTUM:
        return optim.SGD(params, lr=learn_rate, momentum=momentum)
    return optim.SGD(params, lr=learn_rate)


# 自动设备选择
def device():
    if torch.cuda.is_available():
        return torch.device('gpu')
    return torch.device('cpu')
