import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from entity import label


def tensor_image(image):
    """
    普通图像格式转换为tensor数据
    1. 数值转换，/255
    2. 类型转换，float32
    2. 三个通道拆分
    3. 增加一个维度，用以支持模型格式
    """
    image = np.array([list(cv2.split(image))]).astype(np.float32)
    image /= 255
    print(image.shape)
    return torch.from_numpy(image)


def normal_image(image):
    """
    tensor数据转化为一般图片数据
    1. 降维取出
    2. 数据恢复， *255
    3. 格式恢复 uint8
    :param image:
    :return:
    """
    image = image[0]
    image = image * 255
    image = image.numpy().astype(np.uint8)
    return cv2.merge(image)


def load_image(uri):
    bgr = cv2.imdecode(np.fromfile(uri, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def image_show(src):
    _src = src.copy()
    plt.imshow(_src)
    plt.show()


def scale_area(shape, area, scale=0.3):
    """
    截取区域变换
    """
    y, x = shape
    left, up, right, down = area
    scale_x = (right - left) * scale / 2
    scale_y = (down - up) * scale / 2
    left = max(0, left - scale_x)
    right = min(x, right + scale_x)
    up = max(0, up - scale_y)
    down = min(y, down + scale_y)
    return list(map(int, [left, up, right, down]))


def draw_line(src, pt1, pt2):
    """
    划线:
    """
    pt1 = tuple(map(int, pt1))
    pt2 = tuple(map(int, pt2))
    return cv2.line(src, pt1, pt2, color=(255, 0, 0), thickness=3)


def draw_area(src, area):
    """
    画区域
    """
    _src = src.copy()
    left, up, right, down = area
    left_up = (left, up)
    left_down = (left, down)
    right_up = (right, up)
    right_down = (right, down)
    _src = draw_line(_src, left_up, right_up)
    _src = draw_line(_src, left_up, left_down)
    _src = draw_line(_src, right_down, right_up)
    _src = draw_line(_src, right_down, left_down)
    return _src


def draw_point(src, point, color=(255, 0, 0)):
    """
    画点
    """
    point = tuple(map(int, point))
    _src = cv2.circle(src, point, 2, color, thickness=-1)
    return _src


def draw_points(src, points, color=(255, 0, 0)):
    """
    画点集:
    """
    _src = src.copy()
    for point in points:
        _src = draw_point(src, point, color)
    return _src


def split_area(src, area):
    """
    截取指定区域
    """
    area = scale_area(src.shape[:2], area)
    x1, y1, x2, y2 = area
    return src[y1:y2, x1:x2]


def fill_area(src, area, fill=np.array([0, 0, 0])):
    """
    填充指定区域
    """
    x1, y1, x2, y2 = tuple(map(int, area))
    src[y1:y2, x1:x2] = fill
    return src


def resize(src, target_size=(112, 112)):
    return cv2.resize(src, target_size)


def points_scale(shape, points, area, size=(112, 112)):
    """
    点坐标变换
    """
    s_y, s_x = size
    area = scale_area(shape, area)
    x1, y1, x2, y2 = area
    width = x2 - x1
    high = y2 - y1
    points -= np.array([x1, y1])
    points[:, 0] = points[:, 0] * s_x / width
    points[:, 1] = points[:, 1] * s_y / high
    return points


def save_image(_src, name):
    _src = cv2.cvtColor(_src, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, _src)


if __name__ == '__main__':
    with open('../cut/I/label.txt', 'r') as f:
        lines = f.readlines()
    line = lines[889]
    label = label.Label(line)
    image_show(draw_points(label.img, label.points))
