
from torch.hub import load_state_dict_from_url
# from torchvision.models import ResNet
from torchvision import models
from torchvision.models.resnet import BasicBlock
import torchvision.transforms as Transforms
import torch
import torch.nn as nn


from . import ResNet

class ResNet18(ResNet):
  def __init__(self, gamma=0.1):
    self.fcl_input_size = 512
    self.gamma = gamma

    ResNet.__init__(self, block=BasicBlock, layers=[2, 2, 2, 2])
    self.fcl_input_size = self.fc.in_features

    self.dropout = nn.Dropout(0.3)

  def _forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.dropout(x)

    return x

  def _get_fcl_input_size(self):
    return self.fcl_input_size

  def _get_gamma(self):
    return self.gamma
  
def resnet18(pretrained: bool, progress: bool = True, **kwargs):
  """"""
  model = ResNet18(**kwargs)
  if pretrained:
      state_dict = load_state_dict_from_url(models.resnet.model_urls['resnet18'], progress=progress)
      model.load_state_dict(state_dict, strict=False)

  return model

def resnet18_transforms():
  """"""
  return Transforms.Compose([
      Transforms.Resize((80, 80)),
      Transforms.ToTensor(),
      Transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
