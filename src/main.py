# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#
import time
import os
import argparse
import json
import torch.nn as nn
import torch.optim as optim

from datetime import datetime
from torch import torch
from torch import save as save_model
from torch.utils.data import DataLoader
from torchvision import transforms

from verdeolivaNetwork import resnet50nodown
from tuningDatabase import TuningDatabaseFromFile, TuningDatabaseWithRandomSampling
from testing.testingUtils import testModel

# new weights for the model should only affect the fc (fully connected) layer since we are
# doing finetuning
def areNewWeightsValidForFinetuning(model,new_weights):
    areValid = True
    model_dict = model.state_dict()
    for key in model_dict.keys():
        if ("weight" in key) and ("fc" not in key):
            if not (torch.all(model_dict[key].eq(new_weights[key])).item()):
                areValid = False
    
    return areValid

# return a fine-tuned version of the original resnet50 model
# by default num_classes is = 1, since it's specified like that in the original code
def fineTune(model, database, configuration, train_loss_fd,val_loss_fd):
    if configuration["num_classes"] < 2 :
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        model = model.change_output(configuration['num_classes'])
        loss_fn = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=configuration['learning_rate'])

    required_epochs = configuration["epochs"]
    if resume_from_checkpoint:
        checkpoint = torch.load(os.path.join(configuration['checkpoints_path_loading']))
        if(areNewWeightsValidForFinetuning(model,checkpoint['model_state_dict'])):
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print("Error! The weights in the checkpoint are not valid. Aborting...")
            return

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        required_epochs -= checkpoint['epoch']
    elif(configuration["finetuned_weights_to_load"] != None):
        if(areNewWeightsValidForFinetuning(model,torch.load(configuration["finetuned_weights_to_load"]))):
            model.load_state_dict(torch.load(configuration["finetuned_weights_to_load"]))
        else:
            print("Error! The weights in the checkpoint are not valid. Aborting...")
            return

    model = model.to(configuration['device'])
    
    for param in model.parameters():
        param.requires_grad = False;

    model.fc.requires_grad_(True)

    print("Learning rate is: {}".format(configuration["learning_rate"]))
    print("Using {} epochs".format(required_epochs))
    print("Training on parameters:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

    training_start = time.time()
    train_loop(model, database, loss_fn, optimizer, 
                configuration['device'], 
                required_epochs, 
                configuration['perform_validation'], 
                configuration['checkpoints_path_writing'],
                train_loss_fd,val_loss_fd)
    training_end = time.time()

    train_min = (training_end - training_start) // 60
    train_sec = (training_end - training_start) % 60
    print("Training duration: {:.0f} min and {:.0f} seconds".format(train_min,train_sec))

    return model

def train_loop(model, dataloader, loss_fn, optimizer, device, epochs, perform_validation,checkpoints_path, train_loss_fd,val_loss_fd):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    checkpoints_file_name = os.path.join(checkpoints_path,"checkpoint.pth")

    batches_to_accumulate = 1

    for epoch in range(0,epochs):
        print('Epoch {}/{}'.format(epoch,epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        epoch_start_time = time.time()
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # continue to next phase if perform_validation is false
            if (not perform_validation) and phase == 'val':
                continue

            running_corrects = 0
            running_loss = 0
            batch_number = 0
            zerosamples = 0
            zerocorrect = 0
            onecorrect = 0
            onesamples = 0
            for image,target in dataloader[phase]:
                batch_number = batch_number + 1

                with torch.set_grad_enabled(phase=="train"):
                    image = image.to(device)
                    
                    # apply model; we use the same snipped as the one in model.apply, but with grad_enabled since we want
                    # the gradient for backpropagation
                    pred = model(image).to(device)
                    
                    # specify the dimensions to squeeze, or we will destroy the batch dimension, and then 
                    # the loss function may complain since input and output will have different shapes
                    pred_squeezed = torch.squeeze(pred,dim=2)
                    pred_squeezed = torch.squeeze(pred_squeezed,dim=2)

                    # if we do regression (use only 1 output feature), modify the target accordingly
                    if model.fc.out_features == 1:
                        target = target.to(dtype=torch.float32)
                        for index in range(0,target.size()[0]):
                            target[index] = 0.0 if target[index] == 0 else 1.0
                    
                    target = target.to(device)

                    predictions = pred_squeezed.argmax(dim=1, keepdim=True).squeeze()

                    if model.fc.out_features == 1:
                        pred_squeezed = pred_squeezed.squeeze(dim=1)
                    
                    zerosamples += len(target[target==0])
                    zerocorrect += (target[target==predictions]==0).sum().item()
                    onesamples += len(target[target==1])
                    onecorrect += (target[target==predictions]==1).sum().item()

                    correct = (predictions == target).sum().item()
                    accuracy = correct / image.size()[0]        # TODO: maybe a better way to know batch size? 

                    running_corrects += correct

                    loss = loss_fn(pred_squeezed, target)

                    # loss is normalized over accumulation size
                    loss = loss / batches_to_accumulate

                    print("Epoch {}, Batch {} -- {}, Accuracy: {:.4f}, Loss: {:.4f}".format(epoch,batch_number,phase,accuracy,loss.item()))

                    running_loss += loss.item()  * image.size(0)

                    if phase == 'train':
                        loss.backward()
                        
                        if (batch_number % batches_to_accumulate) == 0:
                            optimizer.step()
                            optimizer.zero_grad()

            epoch_acc = running_corrects / len(dataloader[phase].dataset)
            epoch_loss = running_loss / len(dataloader[phase].dataset)
            print("Epoch acc in phase {} : {}".format(phase,epoch_acc))
            print("Epoch loss in phase {} : {}".format(phase,epoch_loss))

            if phase == 'val':
                val_loss_fd.write("{} \n".format(epoch_loss))
                val_loss_fd.flush()
            
            if phase == 'train':
                train_loss_fd.write("{} \n".format(epoch_loss))
                train_loss_fd.flush()
        
        # print epoch duration and estimated time to finish
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        remaining_epochs = epochs - epoch - 1
        current_eta = epoch_duration * remaining_epochs
        
        if not os.path.exists(checkpoints_path) :
            os.makedirs(checkpoints_path)

        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                }, checkpoints_file_name + "_" + str(epoch) + ".pth")
        except OSError:
            print("Could not write checkpoint on epoch {}. Will write the next one.".format(epoch))

        print('Epoch took {:.0f} minutes, {:.0f} seconds.'.format(
                epoch_duration // 60,
                epoch_duration % 60,))
        if remaining_epochs > 0:
            print('Estimated time to finish: {:.0f} minutes,{:.0f} seconds'.format(current_eta // 60,current_eta % 60))
            print('-' * 10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script fine-tunes the original resnet50 model from GANimageDetection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config_path', '-c', type=str, default='config.json')
    parser.add_argument('--weights_path', '-m', type=str, default='./weights/gandetection_resnet50nodown_stylegan2.pth',
                        help='weights path of the network')
    parser.add_argument('--device_to_use', '-d', type=str, default='cuda:0',
                        help='device to use for fine tuning (values: cpu, cuda:0)')
    script_arguments = parser.parse_args()
    
    config_file_fs = open(script_arguments.config_path,"r")
    settings_json = json.loads(config_file_fs.read())

    weights_path = script_arguments.weights_path
    device_to_use = script_arguments.device_to_use
    input_folder = settings_json["DatasetPath"]
    resume_from_checkpoint = settings_json["LoadCheckpoint"]
    batch_size = settings_json["BatchSize"]
    logs_base_path = settings_json["LogsPath"]
    finetuned_weights_filename = settings_json["finetunedWeightsFilename"]

    # generate folder names and paths for the current execution report folder
    today = datetime.now()
    date_folder = today.strftime("%m_%d_%H_%M_%S")
    logs_folder = os.path.join(logs_base_path,date_folder)
    finetuned_weights_path = os.path.join(logs_folder,finetuned_weights_filename)
    checkpoints_path = os.path.join(logs_folder,"checkpoints")

    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    # load dataset and setup dataloaders and transforms (dataloader is used for batching)
    transform_convert_and_normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.Resize([256,256])
    ])

    total_dataset = TuningDatabaseWithRandomSampling(input_folder,transform_convert_and_normalize,seed=13776321,
                                                        real_amount=settings_json["RealSamplesAmount"],
                                                        fake_amount=settings_json["FakeSamplesAmount"])

    for path,_ in total_dataset.samples:
        print(path)
    
    test_catalogue = settings_json["testDatasetCatalogue"]
    val_catalogue = settings_json["validationDatasetCatalogue"]

    test_dataset = TuningDatabaseFromFile(test_catalogue,transform=transform_convert_and_normalize)
    for el in test_dataset.samples:
        if el in total_dataset.samples:
            total_dataset.samples.remove(el)

    validation_dataset = TuningDatabaseFromFile(val_catalogue,transform=transform_convert_and_normalize)
    for el in validation_dataset.samples:
        if el in total_dataset.samples:
            total_dataset.samples.remove(el)

    train_dataset = total_dataset

    databases = {
        'train': train_dataset,
        'testing': test_dataset,
        'val': validation_dataset
    }

    dataloaders = {
        x:DataLoader(databases[x],batch_size=batch_size,shuffle=True) for x in ['train','testing','val']
    }

    # retrieve the data directly from the database, so we do not call the transform on the
    # sample just to read the label
    real_images_count = len(train_dataset.real_images)
    fake_images_count = len(train_dataset.fake_images)
    # TODO: Fix real_images and fake_images count, since right now they do nothing
    print("TRAINING INFO")
    print(f"Using device {device_to_use}")
    print("Dataset located at: {}".format(input_folder))
    print("Will train with {} images and test with {} images".format(len(databases['train']),len(databases['testing'])))
    print("Training dataset composition: \n {} samples labeled real \n {} samples labeled fake".format(real_images_count,fake_images_count))
    print(10*"=")
    print("Training on")
    for filename in train_dataset.samples:
        print(filename)
    print(10*"=")
    print("Validating on")
    for filename in databases['val'].samples:
        print(filename)
    print("Testing on")
    for path,label in databases["testing"].samples:
        print(path)

    settings_json_checkpoint_path = settings_json["CheckpointPath"]

    configuration = {
                "checkpoints_path_loading":settings_json_checkpoint_path,
                "checkpoints_path_writing":checkpoints_path,
                "num_classes":settings_json["Classes"],
                "device":device_to_use,
                "perform_validation":settings_json["PerformValidation"],
                "perform_testing":settings_json["PerformTesting"],
                "learning_rate":settings_json["LearningRate"],
                "epochs":settings_json["Epochs"],
                "finetuned_weights_to_load":None
            }

    validation_loss_history_path = os.path.join(logs_folder,"val_loss_history.log")
    val_loss_fd = open(validation_loss_history_path,"w")

    training_loss_history_path = os.path.join(logs_folder,"train_loss_history.log")
    training_loss_fd = open(training_loss_history_path,"w")

    resnet_no_down_model = resnet50nodown(device_to_use, weights_path)
    print("Training started on {}".format(datetime.now().strftime("%b %a %d at %H:%M:%S")))
    fine_tuned_model = fineTune(resnet_no_down_model,dataloaders,configuration,training_loss_fd,val_loss_fd)
    print("Training ended on {}".format(datetime.now().strftime("%b %a %d at %H:%M:%S")))

    if settings_json["PerformTesting"] :
        testModel(fine_tuned_model,dataloaders,configuration['device'])

    print("Checkpoints are located at {}".format(checkpoints_path))
    print("Saving fine-tuned model (most recent checkpoint) in {}".format(finetuned_weights_path))
    save_model(fine_tuned_model.state_dict(), finetuned_weights_path)