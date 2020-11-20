import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from structures_dataset import StructuresDataset
import torch.nn as nn
from model import get_model
import time
from result_record import make_record_in_csv
import copy
from torch.optim import lr_scheduler

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


def train(model, optimizer, loss, scheduler, train_set, val_set, epochs=1):

    # choose device
    device = get_device()
    print("use device: " + str(device))

    # train loop
    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

        scheduler.step()

        time_per_epoch = time.time() - since
        print("Epoch %d, loss = %.6f, train time = %.4f s" % (epoch, running_loss, time_per_epoch))

        epoch_acc = check_accuracy_in_val(model, val_set)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, "resnet18_weight.pt")
    print("best val acc: %.4f" % best_acc)


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
        num_correct += (preds.data == y).sum()
    acc = float(num_correct) / 5000
    print("total correct predictions is %d, accuracy in val set is %3f" % (num_correct, acc))

    return acc


def test(model, test):

    device = get_device()

    model.eval()

    result = torch.tensor([-1])
    for x in test:
        x = x.to(device)

        scores = model(x)
        _, preds = scores.max(1)
        result = torch.cat((result, preds.cpu()), dim=0)

    return result


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

    # train stage
    learning_rate = 1e-5
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    loss = nn.CrossEntropyLoss()
    train(model, optimizer, loss, exp_lr_scheduler, structures_train, structures_val, 10)

    # test stage
    structures_test = DataLoader(StructuresDataset("test.csv", transform, train=False), batch_size=16,
                                 shuffle=False, drop_last=False, num_workers=4)
    result = test(model, structures_test)

    # record in csv file
    make_record_in_csv(result)


if __name__ == '__main__':
    main()

