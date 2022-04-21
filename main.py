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

import os
import glob
import time
import argparse
from PIL import Image
from resnet50nodown import resnet50nodown
from resnet50fineTuning import resnet50fineTune
from resnet50fineTuning import TuningDatabase
from torch import save as save_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="This script fine-tunes the original resnet50 model from GANimageDetection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--weights_path', '-m', type=str, default='./weights/gandetection_resnet50nodown_stylegan2.pth',
                        help='weights path of the network')
    parser.add_argument('--input_folder', '-i', type=str, default='./example_images',
                        help='input folder with PNG and JPEG images')
    parser.add_argument('--output_csv', '-o', type=str, default=None, help='output CSV file')
    parser.add_argument('--device_to_use', '-d', type=str, default='cuda:0',
                        help='device to use for fine tuning (values: cpu, cuda:0)')
    parser.add_argument('--dry-run', help='just print the options, then exit', action="store_true")
    config = parser.parse_args()
    weights_path = config.weights_path
    input_folder = config.input_folder
    output_csv = config.output_csv
    device_to_use = config.device_to_use
    dry_run = config.dry_run

    from torch.cuda import is_available as is_available_cuda

    # fallback to CPU if CUDA is not available
    device = device_to_use if is_available_cuda() else 'cpu'
    print(f"Using device {device}")
    net = resnet50nodown(device, weights_path)

    if output_csv is None:
        output_csv = 'out.' + os.path.basename(input_folder) + '.csv'

    database = TuningDatabase(input_folder)

    if not dry_run:
        fine_tuned_model = resnet50fineTune(net, database)
        print(f"Saving fine-tuned model")
        save_model(fine_tuned_model.state_dict(), "trained_model_weights.pth")
        print('DONE')
    else:
        print(f"Will use data in {input_folder}")
        print(f"Model will be initialized with weights from {weights_path}")
        print(f"Output will be written in {output_csv}")
        print("Will save model in trained_model_weights.pth")