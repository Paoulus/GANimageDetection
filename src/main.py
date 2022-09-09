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
from datetime import datetime
import os
import glob
import argparse
import json
from PIL import Image
from resnet50nodown import resnet50nodown
from verdeolivaFineTuning import fineTune
from verdeolivaFineTuning import testModel
from torch import utils,arange, tensor
from torch import save as save_model
from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda import is_available as is_available_cuda
from torch.cuda import empty_cache
from torchvision import transforms
from TuningDatabase import TuningDatabaseFromFile, TuningDatabaseWithRandomSampling, TuningDatabaseFromSamples
import csv
import random

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
    
    config_file = open(script_arguments.config_path,"r")
    settings = json.loads(config_file.read())

    weights_path = script_arguments.weights_path
    device_to_use = script_arguments.device_to_use
    input_folder = settings["DatasetPath"]
    resume_from_checkpoint = settings["LoadCheckpoint"]
    batch_size = settings["BatchSize"]
    today = datetime.now()
    date_folder = today.strftime("%m_%d_%H_%M_%S")
    logs_folder = os.path.join("logs",date_folder)
    finetuned_weights_path = os.path.join(logs_folder,"finetuned_weights.pth")
    checkpoints_path = os.path.join(logs_folder,"checkpoints")

    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    total_dataset = TuningDatabaseWithRandomSampling(input_folder,transforms,seed=13776321)

    for path,_ in total_dataset.samples:
        print(path)
    
    test_dataset = TuningDatabaseFromFile("test-set-minuscule.txt")
    for el in test_dataset.samples:
        if el in total_dataset.samples:
            total_dataset.samples.remove(el)

    test_set_size = len(test_dataset.samples)
    validation_set_size = int(test_set_size * 0.2 )
    train_set_size = len(total_dataset.samples)-validation_set_size
    
    train_dataset,validation_dataset = random_split(total_dataset,[train_set_size,validation_set_size])

    databases = {
        'train': train_dataset,
        'testing': test_dataset,
        'val': validation_dataset
    }

    dataloaders = {
        x:DataLoader(databases[x],batch_size=batch_size,shuffle=True) for x in ['train','testing','val']
    }

    # retrieve the data directly from the database, so we do not transform the
    # sample just to read the label
    real_images_count = 0
    fake_images_count = 0
    
    train_dataset_files = []
    for index in databases['train'].indices:
        if total_dataset.samples[index][1] == 0 : 
            real_images_count += 1
        else:
            fake_images_count += 1
        train_dataset_files.append(total_dataset.samples[index][0])

    print("TRAINING INFO")
    print(f"Using device {device_to_use}")
    print("Dataset located at: {}".format(input_folder))
    print("Will train with {} images and test with {} images".format(len(databases['train']),len(databases['testing'])))
    print("Training dataset composition: \n {} samples labeled real \n {} samples labeled fake".format(real_images_count,fake_images_count))
    print(10*"=")
    print("Training on images")
    for filename in train_dataset_files:
        print(filename)
    print(10*"=")
    print("Validating on")
    for index in databases['val'].indices:
        print(total_dataset.samples[index][0])
    print("Testing on")
    for path,label in databases["testing"].samples:
        print(path)

    resnet_no_down_model = resnet50nodown(device_to_use, weights_path)
    print("Training started on {}".format(datetime.now().strftime("%b %a %d at %H:%M:%S")))
    #fine_tuned_model, accuracy_history = fineTune(resnet_no_down_model, dataloaders, device_to_use, settings["Epochs"], settings["LearningRate"], settings["Classes"],resume_from_checkpoint,settings["PerformValidation"],settings["PerformTesting"],checkpoints_path)
    print("Training ended on {}".format(datetime.now().strftime("%b %a %d at %H:%M:%S")))

    testModel(fine_tuned_model,dataloaders,device_to_use)

    print("Checkpoints are located at {}".format(checkpoints_path))
    print("Saving fine-tuned model (most recent checkpoint) in {}".format(finetuned_weights_path))
    save_model(fine_tuned_model.state_dict(), finetuned_weights_path)