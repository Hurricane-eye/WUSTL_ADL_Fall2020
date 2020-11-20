import torch.utils.data as Data
from PIL import Image
import csv
import torch


class StructuresDataset(Data.Dataset):

    def __init__(self, csv_file_path, transform, train=True):
        self.transform = transform
        self.train = train
        self.content = []
        csv_file = open(csv_file_path, "r")
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            self.content.append(line)

    def __getitem__(self, item):
        img_name = self.content[item + 1][0]
        img_path = "./dataset/final-kaggle-data/" + img_name + ".png"
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.train:
            label = torch.tensor(int(self.content[item + 1][1]), dtype=torch.long)
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.content) - 1
