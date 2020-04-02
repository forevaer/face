import torch
from torchvision.transforms import transforms


class ToTensorTransform(object):
    """
    转换为Tensor
    """
    def __call__(self, item):
        """
        由于实体对象的复用，而不是每次返回新的数据，导致数据操作前和操作后类型冲突，应该首先进行类型检查，再决定操作
        """
        # 简单标准化
        item.image = item.image / 255
        if not isinstance(item.image, torch.Tensor):
            item.image = torch.from_numpy(item.image).float()
        if not isinstance(item.points, torch.Tensor):
            item.points = torch.from_numpy(item.points).float()
        if not isinstance(item.label, torch.Tensor):
            item.label = torch.from_numpy(item.label).float()
        item.save()
        return item


merge_transform = transforms.Compose([
    ToTensorTransform()
])
