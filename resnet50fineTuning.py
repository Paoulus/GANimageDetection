import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms
import os
from resnet50nodown import *
from PIL import Image

def train_loop(model,dataloader, loss_fn, optimizer):
    model.train()
    #for index in range(len(dataloader)):
    for index in range(0,1):
        # Compute prediction and loss
        image = dataloader[index]["image"]
        print(image)
        pred = model.evalutate_for_training(image)
        y = torch.tensor(-1.0) if dataloader[index]["label"] == "fake" else torch.tensor(1.0)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class TuningDatabase(datasets.DatasetFolder):
    def __init__(self,path):
        self.classes = ["real","fake"]
        self.file_names = [os.path.join(path,file_name) for file_name in os.listdir(path)]
        pass

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self,idx):
        image = Image.open(self.file_names[idx]).convert('RGB')
        image.load()
        sample = {"label":"fake","image":image}
        return sample

# return a fine-tuned version of the original resnet50 model
def resnet50fineTuning(model,training_set_path):
    model = model.change_output(1)

    optimizer = optim.Adam(model.parameters(),lr=0.0001)     
    loss_fn = nn.BCEWithLogitsLoss()
    data = TuningDatabase(training_set_path)

    train_loop(model,data,loss_fn,optimizer)

    return model