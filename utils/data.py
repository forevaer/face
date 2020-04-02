from entity.dataset import ItemLoader, ItemDataSet
from torch.utils.data import DataLoader
from config import config
from entity.trans import merge_transform


# 加载新格式数据
def get_loader():
    train_txt = '../train.txt'
    test_txt = '../test.txt'
    return DataLoader(ItemDataSet(merge_transform, ItemLoader(train_txt)), config.train_batch_size, shuffle=True),  \
           DataLoader(ItemDataSet(merge_transform, ItemLoader(test_txt)), batch_size=config.test_batch_size)


if __name__ == '__main__':
    train_loader, test_loader = get_loader()
    for _, _ in enumerate(test_loader):
        pass
    for _, _ in enumerate(train_loader):
        pass
