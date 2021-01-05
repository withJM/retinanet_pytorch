import torch
import torch.nn as nn

from resnet import BasicBlock, Bottleneck

### resnet 18, 34, 50, 101, 152 ###
# layers = [[2, 2, 2, 2],
#           [3, 4, 6, 3],
#           [3, 4, 6, 3],
#           [3, 4, 23, 3],
#           [3, 8, 36, 3]]

class RetinaNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 block: BasicBlock or Bottleneck,
                 layers: list):
        super(RetinaNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


