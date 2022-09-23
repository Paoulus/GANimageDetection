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
from torch import utils,arange, tensor
from torch import save as save_model
from torch.utils.data import DataLoader, Subset, random_split
from torch.cuda import is_available as is_available_cuda
from torch.cuda import empty_cache
from torchvision import transforms
from Databases import TuningDatabaseFromFile, TuningDatabaseWithRandomSampling, TuningDatabaseFromSamples
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
    settings_json = json.loads(config_file.read())

    weights_path = script_arguments.weights_path
    device_to_use = script_arguments.device_to_use
    input_folder = settings_json["DatasetPath"]
    resume_from_checkpoint = settings_json["LoadCheckpoint"]
    batch_size = settings_json["BatchSize"]
    
    # generate folder names and paths for the current execution report folder
    today = datetime.now()
    date_folder = today.strftime("%m_%d_%H_%M_%S")
    logs_folder = os.path.join("logs",date_folder)
    finetuned_weights_path = os.path.join(logs_folder,"finetuned_weights.pth")
    checkpoints_path = os.path.join(logs_folder,"checkpoints")

    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    # load dataset and setup dataloaders and transforms (dataloader is used for batching)
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

    # retrieve the data directly from the database, so we do not call the transform on the
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

    configuration = {
                "device_to_use":device_to_use,
                "checkpoints_path":checkpoints_path,
                "num_classes":num_classes,
                "device":device_to_use,
                "perform_validation":perform_validation,
                "perform_testing":perform_testing
                }

    resnet_no_down_model = resnet50nodown(device_to_use, weights_path)
    print("Training started on {}".format(datetime.now().strftime("%b %a %d at %H:%M:%S")))
    fine_tuned_model, accuracy_history = fineTune(resnet_no_down_model,dataloaders,configuration)
    print("Training ended on {}".format(datetime.now().strftime("%b %a %d at %H:%M:%S")))

    testModel(fine_tuned_model,dataloaders,configuration['device_to_use'])

    print("Checkpoints are located at {}".format(checkpoints_path))
    print("Saving fine-tuned model (most recent checkpoint) in {}".format(finetuned_weights_path))
    save_model(fine_tuned_model.state_dict(), finetuned_weights_path)

# return a fine-tuned version of the original resnet50 model
# by default num_classes is = 1, since it's specified like that in the original code
def fineTune(model, database, configuration):
    if num_classes < 2 :
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    checkpoint_epoch = 0
    if resume_from_checkpoint:
        checkpoint = torch.load(os.path.join(configuration['checkpoints_path'],"model_checkpoint_epoch.pt"))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        checkpoint_epoch = checkpoint['epoch']

    for param in model.parameters():
        param.requires_grad = False;

    model = model.change_output(configuration['num_classes'])
    model = model.to(configuration['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=configuration['learning_rate'])

    print("Training on parameters:")
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)
    required_epochs = epochs - checkpoint_epoch

    training_start = time.time()
    validation_history = train_loop(model, database, loss_fn, optimizer, configuration['device'], 
                                                                        required_epochs, 
                                                                        configuration['perform_validation'], 
                                                                        configuration['perform_testing'],
                                                                        configuration['checkpoints_path'])
    training_end = time.time()

    train_min = (training_end - training_start) // 60
    train_sec = (training_end - training_start) % 60
    print("Training duration: {:.0f} min and {:.0f} seconds".format(train_min,train_sec))

    return model, validation_history

def train_loop(model, dataloader, loss_fn, optimizer, device, epochs, perform_validation, perform_testing,checkpoints_path):
    val_loss_history = []
    epoch_acc_history = []

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    checkpoints_file_name = os.path.join(checkpoints_path,"checkpoint.pth")

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
            batch_number = 0
            zerosamples = 0
            zerocorrect = 0
            onecorrect = 0
            onesamples = 0
            for image,target in dataloader[phase]:
                batch_number = batch_number + 1

                with torch.set_grad_enabled(phase=="train"):
                    optimizer.zero_grad()

                    image = image.to(device)
                    # apply model; we use the same snipped as the one in model.apply, but with grad_enabled since we want
                    # the gradient for backpropagation
                    pred = model(image).to(device)
                    # specify the dimensions to squeeze, or we will destroy the batch dimension, and then 
                    # the loss function may complain since input and output will have different shapes
                    pred_squeezed = torch.squeeze(pred,dim=2)
                    pred_squeezed = torch.squeeze(pred_squeezed,dim=2)


                    # if we do regression, modify the target accordingly
                    if model.fc.out_features == 1:
                        target = target.to(dtype=torch.float32)
                        for index in range(0,target.size()[0]):
                            target[index] = 1.0 if target[index] == 0 else -1.0
                    
                    target = target.to(device)

                    predictions = pred_squeezed.argmax(dim=1, keepdim=True).squeeze()
                    
                    zerosamples += len(target[target==0])
                    zerocorrect += (target[target==predictions]==0).sum().item()
                    onesamples += len(target[target==1])
                    onecorrect += (target[target==predictions]==1).sum().item()


                    correct = (predictions == target).sum().item()
                    accuracy = correct / image.size()[0]        # TODO: maybe a better way to know batch size? 

                    running_corrects += correct

                    loss = loss_fn(pred_squeezed, target)

                    print("Epoch {}, Batch {} -- {}, Accuracy: {:.4f}, Loss: {:.4f}".format(epoch,batch_number,phase,accuracy,loss.item()))

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            
                if batch_number % 3 :
                    if not os.path.exists(checkpoints_path) :
                        os.makedirs(checkpoints_path)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.item(),
                        }, checkpoints_file_name)

            epoch_acc = running_corrects / len(dataloader[phase].dataset)
            print("Epoch acc in phase {} : {}".format(phase,epoch_acc))
        
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
            print('-' * 10)
    return epoch_acc_history

def testModel(model,dataloaders,device):
    model.eval()

    zerosamples = 0
    zerocorrect = 0
    onesamples = 0
    onecorrect = 0
    num_correct = 0
    num_samples = 0

    y_true = torch.empty(0, dtype=torch.int).to(device)
    y_pred = torch.empty(0, dtype=torch.int).to(device)

    loader = dataloaders["testing"]

    with torch.no_grad():
        with tqdm(loader, unit="batch") as tbatch:
            for batch_idx, (x, y) in enumerate(tbatch):
                tbatch.set_description(f"Batch {batch_idx}")
                
                x = x.to(device=device)
                y = y.to(device=device)
                y_true = torch.cat((y_true, y))
                
                scores = model(x)

                # since verdeoliva net has a bit of strange outputs, we flatten 
                # them first
                scores = torch.squeeze(scores,dim=2)
                scores = torch.squeeze(scores,dim=2)

                _, predictions = scores.max(1)
                y_pred = torch.cat((y_pred, predictions))

                zerosamples += len(y[y==0])
                zerocorrect += (y[y==predictions]==0).sum().item()
                onesamples += len(y[y==1])
                onecorrect += (y[y==predictions]==1).sum().item()
                num_samples += predictions.size(0)
                num_correct += (predictions == y).sum()
                
                zeroaccuracy = 0
                if zerosamples > 0:
                    zeroaccuracy = float(zerocorrect)/float(zerosamples)*100
                oneaccuracy = 0
                if onesamples > 0:
                    oneaccuracy = float(onecorrect)/float(onesamples)*100
                accuracy = float(num_correct)/float(num_samples)*100
                tbatch.set_postfix(accuracy=accuracy, real=zeroaccuracy, fake=oneaccuracy)

        # Build confusion matrix
        cf_matrix = confusion_matrix(y_true.cpu().numpy(), y_pred.cpu().numpy())
        # do not create dataFrame if there are nan in the cf_matrix
        if cf_matrix.shape == (2,2):
            df_cm = pd.DataFrame((cf_matrix.T/np.sum(cf_matrix,axis=1)).T *100, index = [i for i in ['real','fake']],
                         columns = [i for i in ['real','fake']])
            print('Confusion_Matrix:\n {}\n'.format(df_cm))

        print(f'Got tot: {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f} \n')
        if zerosamples > 0:
            print(f'Got Real: {zerocorrect} / {zerosamples} with accuracy {float(zerocorrect)/float(zerosamples)*100:.2f} \n')
        if onesamples > 0:
            print(f'Got Fake: {onecorrect} / {onesamples} with accuracy {float(onecorrect)/float(onesamples)*100:.2f} \n')

    with open("Acc_Loss.txt", "a") as acclss:
        acclss.write('Acc:{}\n'.format(accuracy))