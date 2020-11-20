import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from structures_dataset import StructuresDataset
import torch.nn as nn
from model import get_model
import time

NUM_TRAIN = 40981
NUM_TEST = 10294
USE_GPU = True
print_every = 1000


def get_device():
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def train(model, optimizer, loss, train_set, val_set, epochs=1):

    # choose device
    device = get_device()
    print("use device: " + str(device))

    # train loop
    model = model.to(device)
    for epoch in range(epochs):

        since = time.time()

        model.train()
        for t, (x, y) in enumerate(train_set):

            x = x.to(device)
            y = y.to(device)

            running_loss = loss(model(x), y)

            optimizer.zero_grad()

            running_loss.backward()

            optimizer.step()

            if t % print_every == 0:
                print("Iteration %d, loss = %.6f" % (t, running_loss))

        time_per_epoch = time.time() - since
        print("Epoch %d, loss = %.6f, train time = %.4f s" % (epoch, running_loss, time_per_epoch))
        check_accuracy_in_val(model, val_set)


def check_accuracy_in_val(model, val):

    device = get_device()

    # check accuracy in val set
    model.eval()
    num_correct = 0
    for x, y in val:
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        _, preds = scores.max(1)
        num_correct += (preds == y).sum()
    acc = float(num_correct) / 5000
    print("total correct prediction is %d, accuracy in val set is %3f" % (num_correct, acc))


def main():
    # building dataset
    transform = T.Compose([
        T.Resize((400, 640)),
        T.ColorJitter(brightness=(1.0, 1.2)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    structures_train = DataLoader(StructuresDataset("train.csv", transform), batch_size=16,
                                  sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)),
                                  drop_last=False, num_workers=4)
    structures_val = DataLoader(StructuresDataset("train.csv", transform), batch_size=16,
                                sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN - 5000, NUM_TRAIN)),
                                drop_last=False, num_workers=4)

    learning_rate = 1e-5

    model = get_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss = nn.CrossEntropyLoss()

    train(model, optimizer, loss, structures_train, structures_val, 10)


if __name__ == '__main__':
    main()

