import numpy as np
from utils import image
import cv2
from torch.utils.data import Dataset


class Item(dict):
    """
    transform操作对象，一般为字典，写成对象方便操作
    注意事项：tensor会多线程进行transform，如果对象复用，数据有时候已经进行转换，transform时候需要增加判断
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.image = kwargs.setdefault('image', None)
        self.label = kwargs.setdefault('label', None)
        self.points = kwargs.setdefault('points', None)
        self.save()

    def save(self):
        """
        数据更新写入
        """
        super().__setitem__('image', self.image)
        super().__setitem__('label', self.label)
        super().__setitem__('points', self.points)


class ItemLoader(object):
    """
    根据文件，自动加载对应数据
    """
    def __init__(self, label_path):
        self.label_path = label_path
        self._items = []
        self.load_items()

    def load_items(self):
        """
        文件加载解析方法
        """
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            info = line.strip().split()
            path = info[0]
            src = np.array(list(cv2.split(image.load_image(path)))).astype(np.float32)
            points = np.array(info[1:]).astype(np.float32)
            label = np.array([0]) if all(points == 0) else np.array([1])
            self._items.append(Item(label=label, image=src, points=points))

    def items(self):
        """
        返回解析元素
        """
        return self._items


class ItemDataSet(Dataset):
    """
    DataSet
    """

    def __init__(self, transform, loader):
        self.transform = transform
        self.items = loader.items()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if self.transform is None:
            return self.items[idx]
        return self.transform(self.items[idx])
