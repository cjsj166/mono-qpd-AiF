import torch
from torch import nn

N = 456



class FeatureConverter(nn.Module):
    def __init__(self, c_res):
        super(FeatureConverter, self).__init__()
        c_h, c_w = c_res

        self.basic_shuffle = nn.PixelShuffle(2)
        self.adapool1 = nn.AdaptiveAvgPool2d((c_h // 4, c_w // 4))
        self.adapool2 = nn.AdaptiveAvgPool2d((c_h // 8, c_w // 8))
        self.adapool3 = nn.AdaptiveAvgPool2d((c_h // 16, c_w // 16))

        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x1, x2, x3 = x
        x1 = self.basic_shuffle(x1)
        x2 = self.basic_shuffle(x2)
        x3 = self.basic_shuffle(x3)

        x1 = self.adapool1(x1)
        x2 = self.adapool2(x2)
        x3 = self.adapool3(x3)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)

        x1 = self.relu1(x1)
        x2 = self.relu2(x2)
        x3 = self.relu3(x3)

        return [x1, x2, x3]

def print_shape(x):
    for i in x:
        print(i.shape)

fc = FeatureConverter((N, N))

x1 = torch.rand(1, 1024, 16, 16)
x2 = torch.rand(1, 1024, 32, 32)
x3 = torch.rand(1, 512, 16, 16)
x = [x1, x2, x3]

print_shape(x)

x = fc(x)
print('')

print_shape(x)


