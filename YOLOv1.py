import torch
import torch.nn as nn

from config import num_classes

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2,stride=2)
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 128, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128,256, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv_block = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            )
        self.conv4 = nn.Sequential(
            self.conv_block,
            self.conv_block,
            self.conv_block,
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1),
        )
        self.Linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 7 * 7 * (self.num_classes + 5)),
            # nn.Sigmoid()
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.Linear(x)
        pred = x.reshape(-1, 7, 7, (self.num_classes + 5))

        pred = pred.view(x.size()[0], -1, (self.num_classes + 5))
        pred_confidence = pred[:, :, :1]
        pred_bboxes = pred[:, :, 1:5]
        pred_classes = pred[:, :, 5:]

        return pred_confidence, pred_classes, pred_bboxes

if __name__ == '__main__':
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    #print(device)

    net = Yolo()
    print(net)
