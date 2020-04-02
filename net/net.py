import torch.nn as nn


class Net(nn.Module):
    """
    主体网络结构
    卷积图片尺寸：O = (I + 2P - K)/S + 1
    """
    def __init__(self):
        super(Net, self).__init__()
        # in_channel, out_channel, kernel_size, stride, padding
        self.conv1_1 = nn.Conv2d(3, 16, 5, 2, 0)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.conv2_1 = nn.Conv2d(16, 32, 3, 1, 0)
        self.bn2_1 = nn.BatchNorm2d(32)
        self.conv2_2 = nn.Conv2d(32, 16, 3, 1, 0)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.bn3_1 = nn.BatchNorm2d(24)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        self.bn3_2 = nn.BatchNorm2d(24)
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(40)
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(80)
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.bn_ip1 = nn.BatchNorm1d(128)
        self.ip2 = nn.Linear(128, 128)
        self.bn_ip2 = nn.BatchNorm1d(128)
        self.ip3 = nn.Linear(128, 42)
        self.conv4_2_cls = nn.Conv2d(40, 40, 3, 1, 1)
        self.bn4_2_cls = nn.BatchNorm2d(40)
        self.ip1_cls = nn.Linear(4 * 4 * 40, 128)
        self.bn_ip1_cls = nn.BatchNorm1d(128)
        self.ip2_cls = nn.Linear(128, 128)
        self.bn_ip2_cls = nn.BatchNorm1d(128)
        self.ip3_cls = nn.Linear(128, 2)
        self.prelu = nn.PReLU()
        self.ave_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # 3->16; （（112 - 5）/ 2 + 1 ）/ 2 =  27
        x = self.ave_pool(self.prelu(self.conv1_1(x)))
        # 16->32; 27 - 3 + 1 = 25
        x = self.prelu(self.conv2_1(x))
        # 32->16; 25 - 3 + 1 = 23
        x = self.prelu(self.conv2_2(x))
        # 16; 23 / 2 = 12
        x = self.ave_pool(x)
        # 16->24; 12 - 3 + 1 = 10
        x = self.prelu(self.conv3_1(x))
        # 24->24; 10 - 3 + 1 = 8
        x = self.prelu(self.conv3_2(x))
        # 24->24; 8 / 2 = 4; 池化不改变通道数
        x = self.ave_pool(x)
        # 24->40; 4 - 3 + 2 + 1 = 4
        x = self.prelu(self.conv4_1(x))
        # 40->80; 4 - 3 + 2 + 1 = 4
        ip3 = self.prelu(self.conv4_2(x))
        # 4 * 4 * 80
        ip3 = ip3.view(-1, 4 * 4 * 80)
        # 4 * 4 * 80->128
        ip3 = self.prelu(self.ip1(ip3))
        # 128 -> 128
        ip3 = self.prelu(self.ip2(ip3))
        # 128 -> 42
        ip3 = self.ip3(ip3)
        # x: 24->40; 4 - 3 + 2 + 1 = 4
        # 40 -> 40; 4 - 3 + 2 + 1 = 4
        ip3_cls = self.prelu(self.conv4_2_cls(x))
        # 4 * 4 * 40
        ip3_cls = ip3_cls.view(-1, 4 * 4 * 40)
        # 4 * 4 * 40 -> 128
        ip3_cls = self.prelu(self.ip1_cls(ip3_cls))
        # 128 -> 128
        ip3_cls = self.prelu(self.ip2_cls(ip3_cls))
        # 128 -> 2
        ip3_cls = self.ip3_cls(ip3_cls)

        return ip3, ip3_cls