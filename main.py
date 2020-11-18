import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from structures_dataset import StructuresDataset
import torch.nn as nn
from model import get_model

NUM_TRAIN = 40981
NUM_TEST = 10294
USE_GPU = False


def get_device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def train(model, optimizer, loss, epochs=1):
    # building dataset
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    structures_train = DataLoader(StructuresDataset("train.csv", transform), batch_size=256,
                                  sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN - 5000)),
                                  drop_last=False)
    structures_val = DataLoader(StructuresDataset("train.csv", transform), batch_size=256,
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN - 5000, NUM_TRAIN)),
                                drop_last=False)

    # choose device
    device = get_device()

    # train loop
    model = model.to(device)
    for epoch in range(epochs):
        for x, y in structures_train:
            x = x.to(device)
            y = y.to(device)

            running_loss = loss(model(x), y)

            optimizer.zero_grad()

            running_loss.backward()

            optimizer.step()
        print("Epoch:%d, loss:%.6f" % (epoch, running_loss))

    # check accuracy in val set
    model.eval()
    num_correct = 0
    for x, y in structures_val:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        mask = scores > 0.5
        preds = scores
        preds[mask] = 1.
        preds *= mask
        num_correct += (preds == y).sum()
    acc = float(num_correct) / 5000
    print("total correct prediction is %d, accuracy in val set is %3f" % (num_correct, acc))


def main():
    learning_rate = 1e-3

    model = get_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = nn.BCELoss()

    train(model, optimizer, loss)


if __name__ == '__main__':
    main()

