from ops import *
from config import config
from config.config import PHASE
from net.net import Net
from utils.data import get_loader
from utils.assist import load_model_dict


# 加载设备
device = config.device()
# 加载模型
model = Net().to(device)
# 加载训练数据
model = load_model_dict(model)
# 获取数据集
train_loader, test_loader = get_loader()
# 分支操作
if config.phase is PHASE.TRAIN:
    train(train_loader, model)
elif config.phase is PHASE.TEST:
    test(test_loader, model)
elif config.phase is PHASE.PREDICT:
    predict(model, config.predict_image)
