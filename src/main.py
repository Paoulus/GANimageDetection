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
import datetime
import os
import glob
from time import  gmtime
import argparse
import json
from PIL import Image
from resnet50nodown import resnet50nodown
from verdeolivaFineTuning import fineTune
from verdeolivaFineTuning import testModel
from verdeolivaFineTuning import TuningDatabase
from torch import utils,arange, tensor
from torch import save as save_model
from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda import is_available as is_available_cuda
from torch.cuda import empty_cache
from torchvision import transforms
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
    config = parser.parse_args()
    
    # by default, use configuration in config.json
    config_file = open(config.config_path,"r")
    settings = json.loads(config_file.read())

    weights_path = config.weights_path
    input_folder = settings["DatasetPath"]
    training_results_path = settings["TrainingResultsPath"]
    device_to_use = config.device_to_use
    resume_from_checkpoint = settings["LoadCheckpoint"]
    checkpoints_path = settings["CheckpointPath"]
    batch_size = settings["BatchSize"]
    finetuned_weights_path = settings["FinetunedWeightsPath"]

    if training_results_path is None:
        training_results_path = 'results.' + os.path.basename(input_folder) + '.csv'

    device = device_to_use if is_available_cuda() else 'cpu'

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    total_dataset = TuningDatabase(input_folder,transforms)

    train_proportion = int(len(total_dataset) * 0.8)
    test_proportion = len(total_dataset) - train_proportion

    train_dataset,testing_dataset = random_split(total_dataset,[train_proportion,test_proportion])

    databases = {
        'train': train_dataset,
        'testing': testing_dataset
    }

    dataloaders = {
        x:DataLoader(databases[x],batch_size=batch_size,shuffle=True) for x in ['train','testing']
    }

    print(f"Using device {device}")
    print("Will train with {} images and test with {} images".format(len(databases['train']),len(databases['testing'])))
    print("Dataset located at: {}".format(input_folder))

    starting_model = resnet50nodown(device, weights_path)
    print("Training started on {}".format(datetime.datetime.now().strftime("%b %a %d at %H:%M:%S")))
    fine_tuned_model, accuracy_history = fineTune(starting_model, dataloaders, device, settings["Epochs"], settings["LearningRate"], settings["Classes"],resume_from_checkpoint,settings["PerformValidation"],settings["PerformTesting"],checkpoints_path)
    print("Training ended on {}".format(datetime.datetime.now().strftime("%b %a %d at %H:%M:%S")))

    testModel(fine_tuned_model,dataloaders,device)

    print("Checkpoints are located at {}".format(checkpoints_path))
    print("Saving fine-tuned model (most recent checkpoint) in {}".format(finetuned_weights_path))
    save_model(fine_tuned_model.state_dict(), finetuned_weights_path)

    with open(training_results_path,"w",newline="") as csvfile:
        fieldnames = ["accuracy","epoch"]
        writer = csv.DictWriter(csvfile,fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(accuracy_history)):
            writer.writerow({"accuracy":accuracy_history[index],"epoch":index})

        empty_cache()

        print('DONE')