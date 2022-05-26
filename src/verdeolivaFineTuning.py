import torch
import torch.nn as nn
from torch.utils.data import RandomSampler
from torch import optim
import torchvision.transforms as transforms
import os
from resnet50nodown import *
from PIL import Image
import copy
import time
from TuningDatabase import *

# return a fine-tuned version of the original resnet50 model
# by default num_classes is = 1, since it's specified like that in the original code
def fineTune(model, database, device, num_classes=1, resume_from_checkpoint=False):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    if num_classes < 2 :
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    starting_epoch = 0
    if resume_from_checkpoint:
        checkpoint = torch.load("./checkpoints/model_checkpoint_epoch.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    for param in model.parameters():
        param.requires_grad = False;

    model = model.change_output(num_classes)
    model.to(device)

    print("Training on parameters:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    training_start = time.time()
    validation_history = train_loop(model, database, loss_fn, optimizer, device,starting_epoch,images_to_use=2,epochs = 2)
    training_end = time.time()

    train_min = (training_end - training_start) // 60
    train_sec = (training_end - training_start) % 60
    print("Training duration: {:.0f} min and {:.0f} seconds".format(train_min,train_sec))

    return model, validation_history


def train_loop(model, dataloader, loss_fn, optimizer, device, starting_epoch,images_to_use=None, epochs = 5):
    best_model_wts = copy.deepcopy(model.state_dict())
    val_loss_history = []
    epoch_acc_history = []
    epoch_losses = []
    epoch_loss = 0

    for epoch in range(starting_epoch,epochs):
        print('Epoch {}/{}'.format(epoch,epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        epoch_start_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            correct_prediction = 0
            if images_to_use is None:
                images_to_use = len(dataloader[phase])

            for image,target in dataloader[phase]:
                # Compute prediction and loss
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()

                    # apply model; we use the same snipped as the one in model.apply, but with grad_enabled since we want
                    # the gradient for backpropagation
                    pred = model(image).to(device)
                    pred_squeezed = torch.squeeze(pred)

                    # if we do regression, modify the target accordingly
                    if model.fc.out_features == 1:
                        target = target.to(dtype=torch.float32)
                        for index in range(0,target.size()[0]):
                            target[index] = 1.0 if target[index] == 0 else -1.0

                    loss = loss_fn(pred_squeezed, target)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    print(loss.item())

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

        # at the end of the epoch, save the obtained model, so that we may be able to resume training if interrupted
        PATH = f"./checkpoints/model_checkpoint_epoch.pt"
        LOSS = 0.4

        if not os.path.exists("./checkpoints/") :
            os.makedirs("./checkpoints/")

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': LOSS,
            }, PATH)

    model.load_state_dict(best_model_wts)
    return epoch_acc_history