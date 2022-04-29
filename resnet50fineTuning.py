import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms
import os
from resnet50nodown import *
from PIL import Image
import copy

def train_loop(model, dataloader, loss_fn, optimizer, device, images_to_use=None, epochs = 5):
    best_model_wts = copy.deepcopy(model.state_dict())
    val_loss_history = []
    epoch_acc_history = []

    for epoch in range(epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            correct_prediction = 0
            incorrect_prediction = 0
            maximum = len(dataloader[phase]) if images_to_use is None else images_to_use;

            for index in range(0, maximum):
                # Compute prediction and loss
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()

                    image, target = dataloader[phase][index]
                    pred = model.evalutate_for_training(image)
                    y = torch.tensor(-1.0) if target == 0 else torch.tensor(1.0)
                    y = y.to(device)
                    loss = loss_fn(pred, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    if (y.item() < 0 and pred.item() < 0) or (y.item() > 0 and pred.item() > 0):
                        correct_prediction += 1

            epoch_loss = running_loss / len(dataloader)
            epoch_acc = correct_prediction / len(dataloader)

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                epoch_acc_history.append(epoch_acc)

    model.load_state_dict(best_model_wts)
    return epoch_acc_history


class TuningDatabase(datasets.DatasetFolder):
    def __init__(self, path):
        self.classes = ["real", "generated"]
        self.samples = []
        for entry in os.listdir(path):
            data_folder = os.path.join(path, entry)
            if entry == 'FFHQ':
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        if file.endswith(".png"):
                            item = os.path.join(root, file), 0
                            self.samples.append(item)
            if entry == 'styleGAN' or entry == 'StyleGAN2':
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        if file.endswith(".png"):
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
def resnet50fineTune(model, database,device):
    # TODO: keep track of running loss?
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.BCEWithLogitsLoss()

    for param in model.parameters():
        param.requires_grad = False;

    model = model.change_output(1)
    model.to(device)

    validation_history = train_loop(model, database, loss_fn, optimizer, device)

    return model, validation_history
