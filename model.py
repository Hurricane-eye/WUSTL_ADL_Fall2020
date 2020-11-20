import torch.nn as nn
import torchvision.models as models


class Flatten(nn.Module):

    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


def get_model():

    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    resnet18.fc = nn.Linear(num_ftrs, 2)

    return resnet18
