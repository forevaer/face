import os
import cv2
import numpy as np
from utils import image


class Label(object):
    """
    初始化元素加载/加工方法
    """
    def __init__(self, line: str, prefix=None, save_path='.'):
        self._line = line
        self.save_path = save_path
        self.prefix = prefix
        self.img = None
        self.path = None
        self.area = None
        self.points = None
        self.scale_point = None
        self.parse()

    def parse(self):
        """
        自动解析原始单行数据
        """
        info = self._line.strip().split()
        if self.prefix is not None:
            self.path = os.path.join(self.prefix, info[0])
        else:
            self.path = info[0]
        self.area = np.array(info[1:5], dtype=np.float)
        self.points = np.array(info[5:], dtype=np.float).reshape(-1, 2)
        try:
            self.img = image.load_image(self.path)
        except:
            raise Exception(f'load image error : {self.path}')

    def image(self, split=False, draw_point=False, draw_area=False, fill=False, resize=False, gray=False):
        """
        图片加工操作
        :param split: 是否根据区域进行裁剪
        :param draw_point: 是否画点
        :param draw_area: 是否画区域
        :param fill: 是否涂黑人脸
        :param resize: 是否进行尺度变换
        :param gray: 是否返回灰度图像
        :return: 返回加工图像
        """
        if not any([split, draw_point, draw_area, fill, resize]):
            return self.img
        _src = self.img
        if draw_point:
            _src = image.draw_points(_src, self.points)
        if draw_area:
            _src = image.draw_area(_src, self.area)
        if fill:
            _src = image.fill_area(_src, self.area)
        if split:
            _src = image.split_area(_src, self.area)
        if resize:
            _src = image.resize(_src)
        if gray:
            _src = cv2.cvtColor(_src, cv2.COLOR_RGB2GRAY)
        return _src

    def point_scale(self):
        """
        裁剪图像之后，对应坐标也需要进行转换
        :return: 返回裁剪并resize之后的变换点坐标
        """
        if self.scale_point is None:
            self.scale_point = image.points_scale(self.img.shape[:2], self.points, self.area)
        return self.scale_point

    def line(self, zero=False):
        """
        转换为单行信息表示，不含框图信息，默认为已经resize的图像
        :param zero: 坐标填充，zero表示填充0，默认为涂黑人脸的坐标生成
        :return: 返回数据行表示
        """
        datas = []
        datas.append(os.path.join(self.save_path, os.path.basename(self.path)))
        if zero:
            datas += np.zeros((42, 1)).astype(np.str_).ravel().tolist()
        else:
            datas += self.point_scale().astype(np.str_).ravel().tolist()
        return ' '.join(datas)


class LabesLoader(object):
    """
    原始数据的批量操作
    """
    def __init__(self, label_path, prefix=None, save_path='.'):
        self.label_path = label_path
        self.save_path = save_path
        self.prefix = prefix
        self.labels = []
        self.load()
        if not os.path.exists(save_path):
            os.makedirs(self.save_path)

    def load(self):
        """
        加载指定原始文件数据
        """
        with open(self.label_path, 'r') as f:
            lines = f.readlines()
        self.labels = [Label(line, self.prefix, self.save_path) for line in lines]

    def items(self):
        """
        返回已加载的原始数据对象列表
        """
        return self.labels

    def save(self, fill=False, split=True):
        """
        存储操作后的图片，并自动生成数据文件
        :param fill: 是否涂黑脸部
        :param split: 是否截取，如果是截取图片，则并应该进行区域截取
        """
        lines = []
        for item in self.labels:
            name = os.path.join(self.save_path, os.path.basename(item.path))
            image.save_image(item.image(split=split, fill=fill, resize=True), name)
            lines.append(item.line(fill is True) + '\n')
        with open(os.path.join(self.save_path, 'label.txt'), 'w+') as f:
            f.writelines(lines)

    def unique(self):
        """
        原始数据中，存在一张图多张人脸，导致数据重复
        生成分类样本的时候，涂黑单张脸还存在其他脸，对于这种图片应该进行过滤
        """
        item_map = {}
        duplicate = []
        for item in self.labels:
            if item.path in duplicate:
                continue
            if item.path in item_map:
                duplicate.append(item.path)
                del item_map[item.path]
            item_map[item.path] = item
        self.labels = item_map.values()
        return self
