import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms
import os
from resnet50nodown import *
from PIL import Image


def train_loop(model, dataloader, loss_fn, optimizer, images_to_use=None):
    maximum = len(dataloader) if images_to_use is None else images_to_use;
    for index in range(0, maximum):
        # Compute prediction and loss
        image, target = dataloader[index]
        pred = model.evalutate_for_training(image)
        y = torch.tensor(-1.0) if target == 0 else torch.tensor(1.0)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TuningDatabase(datasets.DatasetFolder):
    def __init__(self, path):
        self.classes = ["real", "generated"]
        self.samples = []
        for entry in os.listdir(path):
            data_folder = os.path.join(path, entry)
            if entry == 'FFHQ':
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        item = os.path.join(root, file), 0
                        self.samples.append(item)
            if entry == 'styleGAN' or entry == 'StyleGAN2':
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        item = os.path.join(root, file), 1
                        self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def find_classes(self, directory):
        classes_mapping = {"real": 0, "generated": 1}
        return self.classes, classes_mapping

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        return sample, target


# return a fine-tuned version of the original resnet50 model
def resnet50fineTune(model, database):
    # TODO: keep track of running loss?
    model = model.change_output(1)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    train_loop(model, database, loss_fn, optimizer, 1)

    return model
