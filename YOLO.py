import torch
import torch.nn as nn
import torch.nn.functional

from Resnet import resnet18

class YoloBody(nn.Module):
    def __init__(self, num_classes=None):
        super(YoloBody, self).__init__()
        self.num_classes = num_classes  # 类别的数量

        # ----------------backbone: resnet18--------------------
        self.backbone = resnet18(pretrained=True)
        c5 = 512

        # ----------------neck: SPP------------------------------
        self.neck = nn.Sequential(
            SPP(),
            Conv(c5 * 4, c5, k=1),
        )

        # ---------------detection head--------------------------
        self.convsets = nn.Sequential(
            Conv(c5, 256, k=1),
            Conv(256, 512, k=3, p=1),
            Conv(512, 256, k=1),
            Conv(256, 512, k=3, p=1)
        )

        # pred
        self.pred = nn.Conv2d(512, 1 + self.num_classes + 4, kernel_size=1)

    def forward(self, x):
        # backbone主干网络
        c5 = self.backbone(x)

        # neck网络
        p5 = self.neck(c5)

        # detection head网络
        p5 = self.convsets(p5)

        # 预测层
        pred = self.pred(p5)

        # [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        # 理解维度转换，如何对应输出
        pred = pred.view(p5.size(0), 1 + self.num_classes + 4, -1).permute(0, 2, 1)

        # 从预测的pred中分理处objectness、class、txtytwth三部分的预测
        # objectness预测：[B, H*W, 1]
        conf_pred = pred[:, :, :1]
        # class预测：[B, H*W, num_cls]
        cls_pred = pred[:, :, 1: 1 + self.num_classes]
        # bbox预测：[B, H*W, 4]
        txtytwth_pred = pred[:, :, 1 + self.num_classes:]
        return conf_pred, cls_pred, txtytwth_pred


class Conv(nn.Module):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        )

    def forward(self, x):
        return self.convs(x)


class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self, x):
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)

        return x


if __name__ == '__main__':
    # 模型结构测试
    x = torch.randn((1, 3, 416, 416))
    net = YoloBody(num_classes=20)
    y = net(x)
    print(y)
