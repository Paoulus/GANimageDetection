import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torchvision import datasets
from torch import optim
import torchvision.transforms as transforms
import os
from resnet50nodown import *
from PIL import Image
import copy
import time

def train_loop(model, dataloader, loss_fn, optimizer, device, images_to_use=None, epochs = 5):
    best_model_wts = copy.deepcopy(model.state_dict())
    val_loss_history = []
    epoch_acc_history = []

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch,epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        epoch_start_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            best_acc = 0.0
            correct_prediction = 0
            incorrect_prediction = 0
            maximum = len(dataloader[phase]) if images_to_use is None else images_to_use;

            random_sampler = RandomSampler(dataloader[phase],replacement=True,num_samples=images_to_use)

            for index in random_sampler:
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

            epoch_loss = running_loss / len(dataloader[phase])
            epoch_acc = correct_prediction / len(dataloader[phase])

            if phase == 'val':
                val_loss_history.append(epoch_loss)
                epoch_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        # print epoch duration and estimated time to finish
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_epochs = epochs - epoch - 1
        current_eta = epoch_duration * remaining_epochs
        print('Epoch took {:.0f} minutes, {:.0f} seconds.'.format(
                epoch_duration // 60,
                epoch_duration % 60,))
        if remaining_epochs > 0:
            print('Estimated time to finish: {:.0f} minutes,{:.0f} seconds'.format(current_eta // 60,current_eta % 60))
            print('--------------------')

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

    print("Training on parameters:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    training_start = time.time()
    validation_history = train_loop(model, database, loss_fn, optimizer, device,images_to_use=2,epochs = 5)
    training_end = time.time()
    
    train_min = (training_end - training_start) // 60
    train_sec = (training_end - training_start) % 60
    print("Training duration: {:.0f} min and {:.0f} seconds".format(train_min,train_sec))

    return model, validation_history
