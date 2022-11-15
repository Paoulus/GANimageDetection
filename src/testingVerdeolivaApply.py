import argparse
from cgi import test

from main import testModel
from verdeolivaNetwork import *
from TestingDatabase import TestingDatabase
from torch.utils.data import DataLoader
from torchvision import transforms
from Databases import TuningDatabaseWithRandomSampling

batch_size = 2
num_classes = 2

device = "cuda:0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="This script fine-tunes the default pre-trained resnet50 model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_folder', '-i', type=str, default='./fake_dataset',
                        help='input folder with PNG and JPEG images')

    parser.add_argument('--weights_path', '-m', type=str, default='./weights/gandetection_resnet50nodown_stylegan2.pth',
                        help='weights path of the network')

    config = parser.parse_args()
    data_dir = "/media/antoniostefani/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/truebees/forensicsDatasets"
    test_dir = "/home/paolochiste/postsocial-samples/Samples/"
    weights_path = config.weights_path

    starting_model = resnet50nodown(device, weights_path)

    starting_model.fc.to(device)
    starting_model.eval()

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #testing_db = TuningDatabaseWithRandomSampling(data_dir,seed=451)
    testing_db = TestingDatabase(test_dir)

    images_used_for_testing = []

    print("Using 'apply' as a evaluation function")
    for image, label in testing_db:
        image_name = image.filename
        images_used_for_testing.append(image_name)
        assigned_label = "real" if label == 0 else "fake"
        pred = starting_model.apply(image)
        print("{} : {} : {}".format(image_name,assigned_label,pred))


    print("Applying the model directly to the image")
    for image, label in testing_db:
        image_name = image.filename
        assigned_label = "real" if label == 0 else "fake"
        with torch.no_grad():
            pred = starting_model(transforms(image).to(device)[None, :, :, :]).cpu()
        print("{} : {} : {}".format(image_name,assigned_label,pred))

    for image_name in images_used_for_testing:
        print(image_name)
