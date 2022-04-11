import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms
import os
from resnet50nodown import *
from PIL import Image

def train_loop(dataloader, model, loss_fn, optimizer):
    for index in range(len(dataloader)):
        # Compute prediction and loss
        image = dataloader[index]["image"]
        print(image)
        pred = model.apply(image)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 100 == 0:   instead, only do it for first 10 images
        if batch == 10:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            return

class FFTHDatabase(datasets.DatasetFolder):
    def __init__(self,path):
        self.classes = ["real","fake"]
        self.file_names = [os.path.join(path,file_name) for file_name in os.listdir(path)]
        pass

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self,idx):
        image = Image.open(self.file_names[idx]).convert('RGB')
        image.load()
        # TODO: determine label for image from name of file
        sample = {"label":"fake","image":image}
        return sample


def make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        nn.BatchNorm2d(planes * block.expansion),
    )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

# return a fine-tuned version of the original resnet50 model
def resnet50fineTuning(model,training_set_path):
    model.resetLastLayer()
    optimizer = optim.Adam(model.parameters(),lr=0.0001)     
    loss_fn = nn.CrossEntropyLoss()

    data = FFTHDatabase(training_set_path)

    train_loop(data,model,loss_fn,optimizer)

    return model