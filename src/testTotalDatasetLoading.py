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

from pathlib import Path

from verdeolivaNetwork import resnet50nodown
from tuningDatabase import TuningDatabaseFromFile, TuningDatabaseWithRandomSampling
from testing.testingUtils import testModel

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


input_folder = settings_json["datasetPath"]


# load dataset and setup dataloaders and transforms (dataloader is used for batching)
transform_convert_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    #transforms.Resize([256,256])
])

total_dataset = TuningDatabaseWithRandomSampling(input_folder,transform_convert_and_normalize,seed=13776321,
                                                    real_amount=settings_json["realSamplesAmount"],
                                                    fake_amount=settings_json["fakeSamplesAmount"])

print(len(total_dataset))

for filename in total_dataset.samples:
    file_path_object = Path(str(filename[0]))
    file_path_tip = '/'.join(file_path_object.parts[-4:])
    print(file_path_tip)