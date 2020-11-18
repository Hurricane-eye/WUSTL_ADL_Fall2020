import torch.nn as nn


class Flatten(nn.Module):

    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


def get_model():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, 2),
        nn.ReLU(),
        nn.Conv2d(16, 16, 5, 2),
        nn.ReLU(),
        nn.MaxPool2d(5, 1),
        nn.Conv2d(16, 32, 3, 1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(3, 1),
        Flatten(),
        nn.Linear(32 * 43 * 43, 1),
        nn.Sigmoid()
    )

    return model
