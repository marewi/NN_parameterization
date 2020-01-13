import torch
from torch import nn
from torchvision.models import resnet18


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.resnet = resnet18(pretrained=True)

        self.fc_shape = nn.Linear(512, 3)
        self.fc_color = nn.Linear(512, 3)
        self.fc_coord = nn.Sequential(
            nn.Linear(512, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        shape = self.fc_shape(x)
        color = self.fc_color(x)
        coord = self.fc_coord(x)

        return shape, color, coord

if __name__ == '__main__':
    net = Net()
    inp = torch.rand(1,3,128,128)

    output = net(inp)

    for o in output:
        print(o.shape)